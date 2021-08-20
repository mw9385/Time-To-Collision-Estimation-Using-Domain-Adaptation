import time
import board
import busio
import adafruit_lidarlite
import rospy
from std_msgs.msg import Float32

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_lidarlite.LIDARLite(i2c)

if __name__ == '__main__':
    rospy.init_node('Lidar_light')
    pub = rospy.Publisher('depth_sensor', Float32, queue_size = 10)
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        try:
            sensor_measurement = sensor.distance / 100.
            pub.publish(sensor_measurement)
            rate.sleep()
        except RuntimeError as e:
            print(e)

