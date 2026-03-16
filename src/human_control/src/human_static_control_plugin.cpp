/*
 * HumanStaticControlPlugin - Plugin Gazebo 11 per controllare modello umano statico
 * Simile a robot_control_plugin ma per il modello "human" con mesh statica.
 * Sottoscrive /actor/cmd_vel per compatibilità con wasd_controller.py
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>

namespace gazebo
{
  class HumanStaticControlPlugin : public ModelPlugin
  {
  public:
    HumanStaticControlPlugin() : running(true), yaw(0.0) {}

    ~HumanStaticControlPlugin()
    {
      this->running = false;
      if (this->rosThread.joinable())
        this->rosThread.join();
    }

    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Posizione iniziale
      auto pose = this->model->WorldPose();
      this->yaw = pose.Rot().Yaw();

      // ROS init
      if (!ros::isInitialized())
      {
        int argc = 0;
        ros::init(argc, nullptr, "human_static_control_plugin", ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("human_static_control"));

      // Sottoscrivi allo stesso topic dell'actor per compatibilità con wasd_controller
      ros::SubscribeOptions opts = ros::SubscribeOptions::create<geometry_msgs::Twist>(
        "/actor/cmd_vel", 10,
        boost::bind(&HumanStaticControlPlugin::OnCmdVel, this, _1),
        ros::VoidPtr(), &this->rosQueue);
      this->cmdVelSub = this->rosNode->subscribe(opts);

      // Publisher per stato (compatibilità con wasd_controller)
      this->statePub = this->rosNode->advertise<std_msgs::String>("/actor/state", 10);

      this->rosThread = std::thread([this]() {
        ros::Rate rate(100);
        while (this->running && ros::ok()) {
          this->rosQueue.callAvailable(ros::WallDuration(0.01));
          rate.sleep();
        }
      });

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&HumanStaticControlPlugin::OnUpdate, this));

      this->lastTime = this->world->SimTime();
      
      gzmsg << "HumanStaticControlPlugin: Ready! Model: " << _model->GetName() 
            << ", Topic: /actor/cmd_vel" << std::endl;
    }

  private:
    void OnCmdVel(const geometry_msgs::Twist::ConstPtr& msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->cmdVel = *msg;
    }

    void OnUpdate()
    {
      common::Time now = this->world->SimTime();
      double dt = (now - this->lastTime).Double();
      this->lastTime = now;
      if (dt <= 0 || dt > 1.0) return;

      std::lock_guard<std::mutex> lock(this->mutex);

      double vx = this->cmdVel.linear.x;
      double vy = this->cmdVel.linear.y;
      double angularZ = this->cmdVel.angular.z;
      
      bool isMoving = (std::abs(vx) > 0.01 || std::abs(vy) > 0.01 || std::abs(angularZ) > 0.01);
      
      // Pubblica stato
      std_msgs::String stateMsg;
      stateMsg.data = isMoving ? "walking:standing" : "idle:standing";
      this->statePub.publish(stateMsg);

      if (!isMoving) return;

      auto pose = this->model->WorldPose();
      double curX = pose.Pos().X();
      double curY = pose.Pos().Y();
      double curZ = pose.Pos().Z();

      // Calcolo movimento
      // Se vy != 0, si sta muovendo in modalità assoluta (top-down)
      // Se angular.z != 0, si sta ruotando (modalità relativa)
      
      double newX, newY, newYaw;
      
      if (std::abs(angularZ) > 0.01)
      {
        // Modalità relativa: vx è forward, angular.z è rotazione
        this->yaw += angularZ * dt;
        newX = curX + vx * std::cos(this->yaw) * dt;
        newY = curY + vx * std::sin(this->yaw) * dt;
        newYaw = this->yaw;
      }
      else
      {
        // Modalità assoluta: vx è +X, vy è +Y
        newX = curX + vx * dt;
        newY = curY + vy * dt;
        
        // Orienta verso la direzione di movimento solo se movimento significativo
        double speed = std::sqrt(vx*vx + vy*vy);
        if (speed > 0.1)
        {
          // Calcola yaw target
          double targetYaw = std::atan2(vy, vx);
          
          // Interpola smoothly verso il target per evitare scatti
          double yawDiff = targetYaw - this->yaw;
          // Normalizza la differenza in [-pi, pi]
          while (yawDiff > M_PI) yawDiff -= 2.0 * M_PI;
          while (yawDiff < -M_PI) yawDiff += 2.0 * M_PI;
          
          // Smooth rotation (max 5 rad/s)
          double maxYawRate = 5.0;
          if (std::abs(yawDiff) > maxYawRate * dt)
          {
            this->yaw += (yawDiff > 0 ? 1 : -1) * maxYawRate * dt;
          }
          else
          {
            this->yaw = targetYaw;
          }
        }
        newYaw = this->yaw;
      }

      ignition::math::Pose3d newPose(newX, newY, curZ, 0, 0, newYaw);
      this->model->SetWorldPose(newPose);
    }

    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;
    common::Time lastTime;

    double yaw;
    geometry_msgs::Twist cmdVel;
    std::mutex mutex;

    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Subscriber cmdVelSub;
    ros::Publisher statePub;
    ros::CallbackQueue rosQueue;
    std::thread rosThread;
    std::atomic<bool> running;
  };

  GZ_REGISTER_MODEL_PLUGIN(HumanStaticControlPlugin)
}
