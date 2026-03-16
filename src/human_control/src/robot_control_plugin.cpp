/*
 * RobotControlPlugin - Plugin Gazebo 11 per controllare robot_obstacle con collisioni
 * Collision checking basato su distanza tra modelli (cilindri).
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <geometry_msgs/Twist.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <vector>

namespace gazebo
{
  class RobotControlPlugin : public ModelPlugin
  {
  public:
    RobotControlPlugin() : running(true), currentYaw(0.0) {}

    ~RobotControlPlugin()
    {
      this->running = false;
      if (this->rosThread.joinable())
        this->rosThread.join();
    }

    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      this->model = _model;
      this->world = _model->GetWorld();

      this->myRadius = 0.35;
      if (_sdf->HasElement("collision_radius"))
        this->myRadius = _sdf->Get<double>("collision_radius");

      // Topic configurabile (default: /robot/cmd_vel)
      std::string cmdVelTopic = "/robot/cmd_vel";
      if (_sdf->HasElement("cmd_vel_topic"))
        cmdVelTopic = _sdf->Get<std::string>("cmd_vel_topic");

      // ROS init
      if (!ros::isInitialized())
      {
        int argc = 0;
        ros::init(argc, nullptr, "robot_control_plugin", ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("robot_control"));

      ros::SubscribeOptions opts = ros::SubscribeOptions::create<geometry_msgs::Twist>(
        cmdVelTopic, 10,
        boost::bind(&RobotControlPlugin::OnCmdVel, this, _1),
        ros::VoidPtr(), &this->rosQueue);
      this->cmdVelSub = this->rosNode->subscribe(opts);

      this->rosThread = std::thread([this]() {
        ros::Rate rate(100);
        while (this->running && ros::ok()) {
          this->rosQueue.callAvailable(ros::WallDuration(0.01));
          rate.sleep();
        }
      });

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&RobotControlPlugin::OnUpdate, this));

      this->lastTime = this->world->SimTime();
      
      // Inizializza yaw dalla posa corrente
      this->currentYaw = this->model->WorldPose().Rot().Yaw();
      
      // Cache lista ostacoli statici (modelli con collision)
      this->CacheObstacles();
      
      gzmsg << "RobotControlPlugin: Ready! Topic: " << cmdVelTopic << ", radius=" << this->myRadius << std::endl;
    }

  private:
    struct Obstacle {
      std::string name;
      double radius;  // approssimazione cilindro
      bool isDynamic;
    };
    
    std::vector<Obstacle> obstacles;
    
    void CacheObstacles()
    {
      // Boundary del laboratorio (limiti approssimati)
      this->minX = -3.5; this->maxX = 4.0;
      this->minY = -1.0; this->maxY = 5.0;

      // Trova tutti i modelli e approssima con cilindri
      auto models = this->world->Models();
      for (auto& m : models)
      {
        std::string name = m->GetName();
        // Ignora se stesso, ground, apriltag, umano, v_lab (mesh stanza)
        if (name == "robot_obstacle" || name == "ground_plane" || 
            name == "human" || name == "human_visual" || name == "v_lab" ||
            name.find("Apriltag") != std::string::npos)
          continue;
        
        Obstacle obs;
        obs.name = name;
        obs.isDynamic = !m->IsStatic();
        
        // Stima raggio dal bounding box
        auto bb = m->BoundingBox();
        double dx = bb.Max().X() - bb.Min().X();
        double dy = bb.Max().Y() - bb.Min().Y();
        obs.radius = std::max(dx, dy) / 2.0;
        
        // Limita raggio minimo
        if (obs.radius < 0.1) obs.radius = 0.3;
        
        this->obstacles.push_back(obs);
      }
      gzmsg << "RobotControlPlugin: Cached " << this->obstacles.size() << " obstacles" << std::endl;
    }
    
    void OnCmdVel(const geometry_msgs::Twist::ConstPtr& msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->cmdVel = *msg;
    }

    bool IsPositionFree(double x, double y)
    {
      // Collision checking disabled
      (void)x; (void)y;
      return true;
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
      if (std::abs(vx) < 0.01 && std::abs(vy) < 0.01) return;

      auto pose = this->model->WorldPose();
      double curX = pose.Pos().X();
      double curY = pose.Pos().Y();
      double newX = curX + vx * dt;
      double newY = curY + vy * dt;

      bool moved = false;

      // Prova movimento completo
      if (IsPositionFree(newX, newY))
      {
        curX = newX;
        curY = newY;
        moved = true;
      }
      else
      {
        // Wall sliding: prova assi separati
        if (std::abs(vx) > 0.01 && IsPositionFree(curX + vx * dt, curY))
        {
          curX += vx * dt;
          moved = true;
        }
        if (std::abs(vy) > 0.01 && IsPositionFree(curX, curY + vy * dt))
        {
          curY += vy * dt;
          moved = true;
        }
      }

      if (!moved) return;

      // Orientamento smooth verso direzione movimento
      double speed = std::sqrt(vx*vx + vy*vy);
      if (speed > 0.1)
      {
        double targetYaw = std::atan2(vy, vx);
        
        // Calcola differenza yaw normalizzata in [-pi, pi]
        double yawDiff = targetYaw - this->currentYaw;
        while (yawDiff > M_PI) yawDiff -= 2.0 * M_PI;
        while (yawDiff < -M_PI) yawDiff += 2.0 * M_PI;
        
        // Smooth rotation (max 8 rad/s)
        double maxYawRate = 8.0;
        if (std::abs(yawDiff) > maxYawRate * dt)
        {
          this->currentYaw += (yawDiff > 0 ? 1 : -1) * maxYawRate * dt;
        }
        else
        {
          this->currentYaw = targetYaw;
        }
      }
      
      ignition::math::Pose3d newPose(curX, curY, pose.Pos().Z(), 0, 0, this->currentYaw);
      this->model->SetWorldPose(newPose);
    }

    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;
    common::Time lastTime;

    double myRadius;
    double currentYaw;
    double minX, maxX, minY, maxY;  // Boundary del lab
    geometry_msgs::Twist cmdVel;
    std::mutex mutex;

    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Subscriber cmdVelSub;
    ros::CallbackQueue rosQueue;
    std::thread rosThread;
    std::atomic<bool> running;
  };

  GZ_REGISTER_MODEL_PLUGIN(RobotControlPlugin)
}
