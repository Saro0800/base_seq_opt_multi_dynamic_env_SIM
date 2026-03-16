#!/usr/bin/env python

import rospy
import tf
import json
import os
import signal

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tag_transforms.json')

# Global dictionary to store tag transforms
tag_transforms = {}

def save_transforms():
    """Save the current tag transforms to JSON file."""
    if not tag_transforms:
        rospy.logwarn("No tag transforms to save.")
        return
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(tag_transforms, f, indent=4)
        rospy.loginfo(f"Saved {len(tag_transforms)} tag transforms to {OUTPUT_FILE}")
        rospy.loginfo(f"Recorded tags: {sorted(tag_transforms.keys())}")
    except Exception as e:
        rospy.logerr(f"Failed to save tag transforms: {e}")

def shutdown_hook():
    """Called on rospy shutdown."""
    rospy.loginfo("Shutdown requested. Saving latest tag transforms...")
    save_transforms()

def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM to ensure data is saved."""
    rospy.loginfo(f"Signal {sig} received. Saving latest tag transforms...")
    save_transforms()
    rospy.signal_shutdown(f"Signal {sig} received")

def main():
    rospy.init_node('tag_tf_recorder')
    
    # Register shutdown hook and signal handlers
    rospy.on_shutdown(shutdown_hook)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize tf listener
    listener = tf.TransformListener()
    
    # Rate for checking transforms (10 Hz)
    rate = rospy.Rate(10)
    
    # Periodic save interval (seconds)
    save_interval = 5.0
    last_save_time = rospy.get_time()
    
    rospy.loginfo("Tag TF Recorder started. Recording transforms from 'map' to 'tag_*' frames...")
    rospy.loginfo(f"Saving to: {OUTPUT_FILE}")
    
    # Give the listener some time to buffer transforms
    rospy.sleep(1.0)
    
    while not rospy.is_shutdown():
        # Try all possible tag numbers from 0 to 49
        for tag_id in range(50):
            frame_name = f'tag_{tag_id}'
            
            try:
                # Try to get transform from 'map' to this tag frame
                (trans, rot) = listener.lookupTransform('map', frame_name, rospy.Time(0))
                
                # Round to 3 decimal places
                trans_rounded = [round(x, 3) for x in trans]
                rot_rounded = [round(x, 3) for x in rot]
                
                # Store/update in dictionary with the latest values
                tag_transforms[tag_id] = {
                    'position': trans_rounded,
                    'orientation': rot_rounded
                }
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                # Transform not available - tag doesn't exist or can't be reached from 'map'
                pass
        
        # Periodically save to disk to avoid data loss
        current_time = rospy.get_time()
        if current_time - last_save_time >= save_interval:
            save_transforms()
            last_save_time = current_time
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        save_transforms()