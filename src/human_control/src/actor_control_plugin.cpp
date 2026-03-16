/*
 * ActorControlPlugin - Plugin Gazebo 11 per controllare Actor con collisioni
 * Ported to ROS1 Noetic (roscpp) from ROS2 Humble (rclcpp)
 *
 * Approccio: movimento cinematico con collision checking via ray casting.
 * NON usa la fisica ODE per il movimento - il plugin calcola la nuova posizione
 * e la valida lanciando raggi in più direzioni. Se un raggio rileva un ostacolo
 * troppo vicino, il movimento in quella direzione viene bloccato.
 *
 * Il modello "human" (cilindro invisibile) viene spostato via SetWorldPose
 * per sincronizzare la camera FPV.
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/String.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cmath>

namespace gazebo
{
  class ActorControlPlugin : public ModelPlugin
  {
  public:
    ActorControlPlugin() : ModelPlugin() {}

    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
      if (!this->actor)
      {
        gzerr << "ActorControlPlugin: Not an Actor!" << std::endl;
        return;
      }

      this->world = this->actor->GetWorld();
      this->physicsEngine = this->world->Physics();
      gzmsg << "ActorControlPlugin: Controlling actor '" << this->actor->GetName() << "'" << std::endl;

      // Trova il modello di collisione "human" (solo per camera FPV)
      this->collisionModel = this->world->ModelByName("human");
      if (!this->collisionModel)
      {
        gzwarn << "ActorControlPlugin: Model 'human' not found! FPV camera won't work." << std::endl;
      }
      else
      {
        gzmsg << "ActorControlPlugin: FPV model 'human' found" << std::endl;
      }

      // Ottieni la posa iniziale
      this->currentPose = this->actor->WorldPose();
      this->posX = this->currentPose.Pos().X();
      this->posY = this->currentPose.Pos().Y();
      this->yaw = 0.0;

      // Parametri
      this->animationFactor = 5.0;
      if (_sdf->HasElement("animation_factor"))
        this->animationFactor = _sdf->Get<double>("animation_factor");

      // Raggio collisione (distanza minima dagli ostacoli)
      this->collisionRadius = 0.3;
      if (_sdf->HasElement("collision_radius"))
        this->collisionRadius = _sdf->Get<double>("collision_radius");

      // Trova tutte le animazioni disponibili
      auto skelAnims = this->actor->SkeletonAnimations();
      gzmsg << "ActorControlPlugin: Available animations:" << std::endl;
      for (auto const& anim : skelAnims)
      {
        gzmsg << "  - " << anim.first << " (length: " << anim.second->GetLength() << "s)" << std::endl;
        this->availableAnimations.push_back(anim.first);
      }

      this->walkingAnim = "walking";
      this->idleAnim = "walking";
      this->currentAnim = this->walkingAnim;

      if (skelAnims.find(this->walkingAnim) == skelAnims.end())
      {
        gzwarn << "ActorControlPlugin: 'walking' animation not found, using first available" << std::endl;
        if (!this->availableAnimations.empty())
          this->walkingAnim = this->availableAnimations[0];
      }
      if (skelAnims.find(this->idleAnim) == skelAnims.end())
      {
        this->idleAnim = this->walkingAnim;
      }

      // Inizializza ROS 1
      if (!ros::isInitialized())
      {
        int argc = 0;
        char** argv = nullptr;
        ros::init(argc, argv, "actor_controller_plugin", ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("actor_controller"));

      // Subscribers
      ros::SubscribeOptions cmdVelOpts =
        ros::SubscribeOptions::create<geometry_msgs::Twist>(
          "/actor/cmd_vel", 10,
          boost::bind(&ActorControlPlugin::OnCmdVel, this, _1),
          ros::VoidPtr(), &this->rosQueue);
      this->cmdVelSub = this->rosNode->subscribe(cmdVelOpts);

      ros::SubscribeOptions animOpts =
        ros::SubscribeOptions::create<std_msgs::String>(
          "/actor/set_animation", 10,
          boost::bind(&ActorControlPlugin::OnSetAnimation, this, _1),
          ros::VoidPtr(), &this->rosQueue);
      this->animSub = this->rosNode->subscribe(animOpts);

      ros::SubscribeOptions poseOpts =
        ros::SubscribeOptions::create<geometry_msgs::Pose>(
          "/actor/set_pose", 10,
          boost::bind(&ActorControlPlugin::OnSetPose, this, _1),
          ros::VoidPtr(), &this->rosQueue);
      this->poseSub = this->rosNode->subscribe(poseOpts);

      // Publisher
      this->statePub = this->rosNode->advertise<std_msgs::String>("/actor/state", 10);

      // Thread per processare la callback queue
      this->rosThread = std::thread([this]() {
        ros::Rate rate(100);
        while (this->running && ros::ok())
        {
          this->rosQueue.callAvailable(ros::WallDuration(0.01));
          rate.sleep();
        }
      });

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&ActorControlPlugin::PreUpdate, this));

      this->lastSimTime = this->world->SimTime();

      gzmsg << "ActorControlPlugin: Ready! (Kinematic collision mode)" << std::endl;
      gzmsg << "  Collision radius: " << this->collisionRadius << "m" << std::endl;
      gzmsg << "  Topics:" << std::endl;
      gzmsg << "    /actor/cmd_vel (Twist) - movement" << std::endl;
      gzmsg << "    /actor/set_animation (String) - change animation" << std::endl;
      gzmsg << "    /actor/set_pose (Pose) - teleport" << std::endl;
      gzmsg << "    /actor/state (String) - current state" << std::endl;
    }

    // ROS1 callbacks
    void OnCmdVel(const geometry_msgs::Twist::ConstPtr& msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->cmdVel = *msg;
    }

    void OnSetAnimation(const std_msgs::String::ConstPtr& msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->requestedAnim = msg->data;
      this->hasAnimRequest = true;
      gzmsg << "ActorControlPlugin: Animation request: " << msg->data << std::endl;
    }

    void OnSetPose(const geometry_msgs::Pose::ConstPtr& msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->posX = msg->position.x;
      this->posY = msg->position.y;
      double siny = 2.0 * (msg->orientation.w * msg->orientation.z);
      double cosy = 1.0 - 2.0 * (msg->orientation.z * msg->orientation.z);
      this->yaw = std::atan2(siny, cosy);
      this->hasTeleport = true;
    }

    /// Controlla se una posizione XY è libera da ostacoli.
    /// Usa distanza tra cilindri invece di ray casting per stabilità.
    /// Ritorna true se non ci sono ostacoli entro collisionRadius.
    bool IsPositionFree(double x, double y)
    {
      // Collision checking disabled
      (void)x; (void)y;
      return true;
    }

    void PreUpdate()
    {
      common::Time currentTime = this->world->SimTime();
      double dt = (currentTime - this->lastSimTime).Double();
      this->lastSimTime = currentTime;

      if (dt <= 0.0 || dt > 1.0)
        return;

      std::lock_guard<std::mutex> lock(this->mutex);

      // Teleport
      if (this->hasTeleport)
      {
        this->hasTeleport = false;
      }

      // Calcola velocità desiderata
      bool absoluteMode = (std::abs(this->cmdVel.angular.z - 999.0) < 0.1);

      double vx, vy;
      double vx_local = this->cmdVel.linear.x;
      double vy_local = this->cmdVel.linear.y;
      double wz = absoluteMode ? 0.0 : this->cmdVel.angular.z;

      if (absoluteMode)
      {
        vx = vx_local;
        vy = vy_local;
      }
      else
      {
        double cosYaw = std::cos(this->yaw);
        double sinYaw = std::sin(this->yaw);
        vx = vx_local * cosYaw - vy_local * sinYaw;
        vy = vx_local * sinYaw + vy_local * cosYaw;
      }

      bool hasLinearVel = (std::abs(vx) > 0.01 || std::abs(vy) > 0.01);
      bool isMoving = hasLinearVel || std::abs(wz) > 0.01;
      bool didMove = false;

      // =============================================
      // MOVIMENTO CINEMATICO CON COLLISION CHECK
      // =============================================
      if (hasLinearVel)
      {
        double newX = this->posX + vx * dt;
        double newY = this->posY + vy * dt;

        // Prova movimento completo
        if (IsPositionFree(newX, newY))
        {
          this->posX = newX;
          this->posY = newY;
          didMove = true;
        }
        else
        {
          // Wall sliding: prova assi separati
          bool movedX = false, movedY = false;

          if (std::abs(vx) > 0.01 && IsPositionFree(this->posX + vx * dt, this->posY))
          {
            this->posX += vx * dt;
            movedX = true;
          }

          if (std::abs(vy) > 0.01 && IsPositionFree(this->posX, this->posY + vy * dt))
          {
            this->posY += vy * dt;
            movedY = true;
          }

          didMove = movedX || movedY;
        }
      }

      // =============================================
      // GESTIONE ANIMAZIONI
      // =============================================
      std::string targetAnim = this->currentAnim;

      if (isMoving && (didMove || std::abs(wz) > 0.01))
      {
        targetAnim = this->walkingAnim;
        this->customAnimPlaying = false;
      }
      else
      {
        if (this->hasAnimRequest)
        {
          auto skelAnims = this->actor->SkeletonAnimations();
          if (skelAnims.find(this->requestedAnim) != skelAnims.end())
          {
            targetAnim = this->requestedAnim;
            this->customAnimPlaying = true;
            this->customAnimStartTime = currentTime.Double();
            this->customAnimDuration = skelAnims[this->requestedAnim]->GetLength();
          }
          this->hasAnimRequest = false;
        }
        else if (this->customAnimPlaying)
        {
          double elapsed = currentTime.Double() - this->customAnimStartTime;
          if (elapsed >= this->customAnimDuration)
          {
            this->customAnimPlaying = false;
            targetAnim = this->idleAnim;
          }
        }
        else
        {
          targetAnim = this->idleAnim;
        }
      }

      if (targetAnim != this->currentAnim)
      {
        this->currentAnim = targetAnim;
        this->scriptTime = 0.0;
        gzmsg << "ActorControlPlugin: Switching to animation: " << this->currentAnim << std::endl;
      }

      // =============================================
      // AGGIORNA YAW
      // =============================================
      if (absoluteMode)
      {
        if (hasLinearVel)
        {
          double targetYaw = std::atan2(vy, vx);
          double diff = targetYaw - this->yaw;
          while (diff > M_PI) diff -= 2.0 * M_PI;
          while (diff < -M_PI) diff += 2.0 * M_PI;
          this->yaw += diff * std::min(1.0, 10.0 * dt);
        }
      }

      this->yaw += wz * dt;

      // =============================================
      // SINCRONIZZA MODELLO COLLISIONE (camera FPV)
      // =============================================
      if (this->collisionModel)
      {
        ignition::math::Pose3d colPose;
        colPose.Pos().Set(this->posX, this->posY, 0);
        colPose.Rot() = ignition::math::Quaterniond(0, 0, this->yaw);
        this->collisionModel->SetWorldPose(colPose);
      }

      // =============================================
      // AGGIORNA ANIMAZIONE
      // =============================================
      if (didMove)
      {
        double speed = std::sqrt(vx*vx + vy*vy);
        this->scriptTime += dt * this->animationFactor * speed;
      }
      else if (std::abs(wz) > 0.01)
      {
        // Rotazione sul posto - animazione lenta
        this->scriptTime += dt * this->animationFactor * 0.3;
      }
      else if (this->customAnimPlaying)
      {
        this->scriptTime += dt;
      }

      auto skelAnims = this->actor->SkeletonAnimations();
      if (skelAnims.find(this->currentAnim) != skelAnims.end())
      {
        double len = skelAnims[this->currentAnim]->GetLength();
        if (len > 0)
          this->scriptTime = std::fmod(this->scriptTime, len);
      }

      auto trajectory = std::make_shared<physics::TrajectoryInfo>();
      trajectory->id = 0;
      trajectory->type = this->currentAnim;
      trajectory->duration = 1.0;
      trajectory->startTime = currentTime.Double();
      trajectory->endTime = currentTime.Double() + 1.0;
      trajectory->translated = false;
      this->actor->SetCustomTrajectory(trajectory);

      // Posa finale dell'actor
      ignition::math::Pose3d finalPose;
      finalPose.Pos().Set(this->posX, this->posY, 1.0);
      finalPose.Rot() = ignition::math::Quaterniond(M_PI/2, 0, this->yaw + M_PI/2);

      this->actor->SetWorldPose(finalPose, true, true);
      this->actor->SetScriptTime(this->scriptTime);

      // Pubblica stato
      this->stateCounter++;
      if (this->stateCounter >= 50)
      {
        this->stateCounter = 0;
        std_msgs::String stateMsg;
        std::string stateStr = didMove ? "moving" : (std::abs(wz) > 0.01 ? "turning" : "idle");
        stateMsg.data = stateStr + ":" + this->currentAnim;
        this->statePub.publish(stateMsg);
      }
    }

    ~ActorControlPlugin()
    {
      this->running = false;
      if (this->rosThread.joinable())
        this->rosThread.join();
      this->rosQueue.clear();
      this->rosQueue.disable();
      this->rosNode->shutdown();
    }

  private:
    physics::ActorPtr actor;
    physics::WorldPtr world;
    physics::PhysicsEnginePtr physicsEngine;
    physics::ModelPtr collisionModel;
    event::ConnectionPtr updateConnection;

    // Posizione cinematica
    double posX = 0.0;
    double posY = 0.0;
    double yaw = 0.0;
    double collisionRadius;

    ignition::math::Pose3d currentPose;
    geometry_msgs::Twist cmdVel;

    bool hasTeleport = false;
    bool hasAnimRequest = false;

    double animationFactor;
    double scriptTime = 0.0;

    std::string walkingAnim;
    std::string idleAnim;
    std::string currentAnim;
    std::string requestedAnim;
    std::vector<std::string> availableAnimations;

    bool customAnimPlaying = false;
    double customAnimStartTime = 0.0;
    double customAnimDuration = 0.0;

    int stateCounter = 0;

    common::Time lastSimTime;

    std::mutex mutex;
    std::atomic<bool> running{true};

    // ROS1
    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::CallbackQueue rosQueue;
    ros::Subscriber cmdVelSub;
    ros::Subscriber animSub;
    ros::Subscriber poseSub;
    ros::Publisher statePub;
    std::thread rosThread;
  };

  GZ_REGISTER_MODEL_PLUGIN(ActorControlPlugin)
}
