#!/usr/bin/python
# Copyright 2020 Sriharsha Bhat (svbhat@kth.se)
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function

import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
import rospy

import tf
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles, PercentStamped
from std_msgs.msg import Float64, Header, Bool
import math

from tf.transformations import quaternion_from_euler, euler_from_quaternion


class MPC_Controller(object):

    # Skew function for dynamics model
    def skew(self, l):
        l = l.flatten()
        l1 = l[0]
        l2 = l[1]
        l3 = l[2]
        return np.array([[0, -l3, l2], [l3, 0, -l1], [-l2, l1, 0]])

    # Nonlinear dynamics model function
    def eom(self, state, control):
        # extract states and controls
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        rpm1, rpm2, de, dr, lcg, vbs = control
        #de, dr = control
        #rpm1,rpm2,de,dr = control

        eta = np.array([[x], [y], [z], [phi], [theta], [psi]])
        nu = np.array([[u], [v], [w], [p], [q], [r]])

        # scaling controls
        rpm_scale = 500.
        d_scale = 0.1
        vbs_scale = 1.
        lcg_scale = 1.
        
        rpm1 = rpm1 * rpm_scale
        rpm2 = rpm2 * rpm_scale
        de = de * d_scale
        dr = dr * d_scale
        vbs = vbs * vbs_scale
        lcg = lcg * lcg_scale
        

        # assign parameters
        m = 15.4  # mass
        Ixx = 0.0294 
        Iyy = 1.6202
        Izz = 1.6202 
        I_o = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

        # cg position
        x_g = 0.0 + lcg*0.01
        y_g = 0.0
        z_g = 0.0
        r_g = np.array([x_g, y_g, z_g])

        # cb position
        x_b = 0.0
        y_b = 0.0
        z_b = 0.0
        r_b = np.array([x_b, y_b, z_b])

        # center of pressure position
        x_cp = 0.1
        y_cp = 0.0
        z_cp = 0.0
        r_cp = np.array([x_cp, y_cp, z_cp])

        W = m * 9.81
        B = W + vbs* 1.5

        # Hydrodynamic coefficients
        Xuu = 3. #1.0
        Yvv = 50. #100.0
        Zww = 50. #100.0
        Kpp = 0.1 #10.0
        Mqq = 40 #100.0
        Nrr = 40 #150.0

        # Control actuators
        K_T = np.array([0.0175, 0.0175])
        Q_T = np.array([0.001, -0.001])

        # Mass and inertia matrix
        M = np.block([[m * np.eye(3, 3), -m * self.skew(r_g)], [m * self.skew(r_g), I_o]])
        assert M.shape == (6, 6), M

        # Coriolis and centripetal matrix
        nu1 = np.array([[u], [v], [w]])
        nu2 = np.array([[p], [q], [r]])
        top_right = -m * self.skew(nu1) - m * self.skew(nu2) * self.skew(r_g)
        bottom_left = -m * self.skew(nu1) + m * self.skew(r_g) * self.skew(nu2)
        Ionu2 = I_o.dot(nu2)
        bottom_right = -self.skew(Ionu2)
        C_RB = np.block([[np.zeros([3, 3]), top_right], [bottom_left, bottom_right]])
        assert C_RB.shape == (6, 6), C_RB

        # Damping matrix
        Forces = np.diag([Xuu * abs(u), Yvv * abs(v), Zww * abs(w)])
        Moments = np.diag([Kpp * abs(p), Mqq * abs(q), Nrr * abs(r)])
        Coupling = np.matmul(self.skew(r_cp), Forces)
        D = np.block([[Forces, np.zeros([3, 3])], [-Coupling, Moments]])
        assert D.shape == (6, 6), D

        # rotational transform between body and NED in Euler        
        T_euler = np.array(
            [
                [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
            ]
        )

        R_euler = np.array(
            [
                [
                    np.cos(psi)*np.cos(theta),
                    -np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi),
                    np.sin(psi)*np.sin(phi)+np.cos(psi)*np.cos(phi)*np.sin(theta),
                ],
                [
                    np.sin(psi)*np.cos(theta),
                    np.cos(psi)*np.cos(phi)+np.sin(phi)*np.sin(theta)*np.sin(psi),
                    -np.cos(psi)*np.sin(phi)+np.sin(theta)*np.sin(psi)*np.cos(phi),
                ],
                [
                    -np.sin(theta),
                    np.cos(theta)*np.sin(phi),
                    np.cos(theta)*np.cos(phi),
                ],
            ]
        )
        assert R_euler.shape == (3, 3), R_euler

        J_eta = np.block([[R_euler, np.zeros([3, 3])], [np.zeros([3, 3]), T_euler]])
        assert J_eta.shape == (6, 6), J_eta

        # buoyancy in quaternions
        f_g = np.array([[0], [0], [W]])
        f_b = np.array([[0], [0], [-B]])
        row1 = [np.linalg.inv(R_euler).dot((f_g + f_b))]
        row2 = [
            self.skew(r_g).dot(np.linalg.inv(R_euler)).dot(f_g)
            + self.skew(r_b).dot(np.linalg.inv(R_euler)).dot(f_b)
        ]
        geta = np.block([row1, row2])
        assert geta.shape == (6, 1), geta


        F_T = K_T.dot(np.array([[rpm1], [rpm2]]))
        M_T = Q_T.dot(np.array([[rpm1], [rpm2]]))
        tauc = np.block(
            [
                [F_T * np.cos(de) * np.cos(dr)],
                [-F_T * np.sin(dr)],
                [F_T * np.sin(de) * np.cos(dr)],
                [M_T * np.cos(de) * np.cos(dr)],
                [-M_T * np.sin(dr)],
                [M_T * np.sin(de) * np.cos(dr)],
            ]
        )
        assert tauc.shape == (6, 1), tauc
        # Kinematics
        etadot = np.block([J_eta.dot(nu)])

        assert etadot.shape == (6, 1), etadot

        # Dynamics
        invM = np.linalg.inv(M)
        nugeta = nu - geta
        crbd = C_RB + D
        other = crbd.dot(nugeta)
        other2 = tauc - other
        nudot = invM.dot(other2)

        assert nudot.shape == (6, 1), nudot
        sdot = np.block([[etadot], [nudot]])

        return sdot.flatten()

    # Linearizing the nonlinear model and output function to obtain the A, B and C matrices. The D matrix is a set of zeros so we do not use it.
    # Numerical Jacobian function A Matrix
    def jacA(self, state, control):

        eps = 1e-10
        J = np.zeros([len(state), len(state)], dtype=np.float)

        for i in range(len(state)):
            x1 = state.copy()
            x2 = state.copy()

            x1[i] += eps
            x2[i] -= eps

            f1 = self.eom(x1, control)
            f2 = self.eom(x2, control)

            J[:, i] = 1 / (2 * eps) * (f1 - f2)

        return J

    # Numerical Jacobian function B Matrix
    def jacB(self, state, control):

        eps = 1e-10
        J = np.zeros([len(state), len(control)], dtype=np.float)

        for i in range(len(control)):
            control1 = control.copy()
            control2 = control.copy()

            control1[i] += eps
            control2[i] -= eps

            f1 = self.eom(state, control1)
            f2 = self.eom(state, control2)

            J[:, i] = 1 / (2 * eps) * (f1 - f2)

        return J



    # Function to obtain linear model at each time step
    def getLinearModel(self, state, control):
        A_jac = self.jacA(state, control)
        B_jac = self.jacB(state, control)
        return [A_jac, B_jac]

    # Function to call the optimal controller using CVX at each time step
    def MPCmove(self, A, B, x_0, u_prev, x_ref, N):
        # epsi = 10e-5 #adding a tiny term to make the linearization numerically feasible and not singular or too sparse making the solver not converge
        #x_0[:] = x_0[:]  # +  epsi
        # u_prev[:]= u_prev[:] + epsi # if you consider previous control- on hold, see how it goes without it first

        N = 3 # Prediction horizon
        [nx, nu] = B.shape  # size of state and control vectors
        A = (
            np.eye(nx, nx) + A
        )  # converting continuous model to discrete time by adding an identity matrix

        xk = cp.Variable((nx, N + 1))  # states including euler pose
        uk = cp.Variable((nu, N))
        Q = 1000 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Weighting matrix on states [13 states]
        QN = 1000 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #weight on final state
        R = 10 * np.eye(nu, nu) # weights on controls
        #R = np.diag([10,10,10,10,10,10]) # weights on controls [rpm1, rpm2, de, dr]

        # constraints on states and inputs
        #umin = np.array([-5.0, -5.0, -15, -15])
        #umax = np.array([5.0, 5.0, 15, 15])
        umin = -1*np.ones(nu)
        umax = 1*np.ones(nu)

        cost = 0
        constr = []
        for k in range(N):
            cost += cp.quad_form(xk[:, k + 1] - x_ref, Q) + cp.quad_form(uk[:, k], R)
            #constr += [
            #    xk[:, k + 1] == A @ xk[:, k] + B @ uk[:, k],
            #    yk[:, k] == C @ xk[:, k],
            #    xk[:, 0] == x_0,
            #    umin <= uk[:, k],
            #    uk[:, k] <= umax,
            #]
            #Python2 
            constr += [
                xk[:, k + 1] == A * xk[:, k] + B * uk[:, k],
                xk[:, 0] == x_0,
                umin <= uk[:, k],
                uk[:, k] <= umax,
            ]
            # uk[:,0] == u_prev, #consider if you want to use the previous control
        # sums problem objectives and concatenates constraints.
        # constr += [x[:,T] == 0, x[:,0] == x_0]
        cost += cp.quad_form(xk[:, N] - x_ref, QN)
        problem = cp.Problem(cp.Minimize(cost), constr)
        # problem.solve(solver=cp.ECOS)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # print(uk[:,0].value)
        #scale the control
        ctrl_scale = np.diag([500,500,0.1,0.1, 1., 1.])
        #ctrl_scale = np.diag([ 0.1, 0.1])
        u_computed = ctrl_scale.dot(uk[:, 0].value) 

        #return uk[:, 0].value
        return u_computed

    #Function to test if the controller and solver and models work!
    '''def testMPC(self):
            x_1 = np.array([10, 10, -3, 1, 0, 0, 0, 1, 0.1, 0.1, 0.01, 0.01, 0.01])
            u_1 = np.array([100, 100, 0.01, 0.01])
            [A, B, C] = self.getLinearModel(x_1, u_1)
            yref = np.array([0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # reference values
            N = 20
            ctrl = self.MPCmove(A, B, C, x_1, u_1, yref, N)
            print("Testng done! :)  Optimal ctrl=", ctrl)'''

    '''def dummy_setpoint(self):
        x= 30.
        y= 6.
        z= -6.
        phi = 0.
        theta = 0.
        psi = 0.
        u = 0.
        v = 0.
        w = 0.
        p = 0.
        q = 0.
        r = 0.
        y_out= np.array([x,y,z,phi,theta,psi,u,v,w,p,q,r])
        return y_out'''

    def computeMPC(self):
        #print('computing control action based on feedback')
        
        if self.counter % 1  == 0:     
            next_u = self.MPCmove(self.A_lin, self.B_lin, self.current_x, self.current_u, self.current_setpoint, self.prediction_horizon)

        if self.counter % 5 == 0:
            # linearize periodically after adding a small additive term to reduce numerical instabilities
            self.current_x[:] = self.current_x[:] + self.epsi
            self.current_u[:] = self.current_u[:] + self.epsi
            [self.A_lin, self.B_lin] = self.getLinearModel(self.current_x, self.current_u)
            #print('Relinearizing!!', self.counter)

            self.current_u = next_u # assign optimal control
        
        self.counter = self.counter + 1
        #return next_u

    # Function to subscribe to feedback topics
    def get_state_feedback(self, odom_msg):
        # Insert functon to subscribe from \ODOM here
        #print('getting state feedback')
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z
        eta0 = odom_msg.pose.pose.orientation.w
        eps1 = odom_msg.pose.pose.orientation.x
        eps2 = odom_msg.pose.pose.orientation.y
        eps3 = odom_msg.pose.pose.orientation.z

        rpy= euler_from_quaternion([eps1,eps2,eps3,eta0])
        roll = rpy[0]
        pitch = -rpy[1]
        yaw = -rpy[2]

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z        
        p= odom_msg.twist.twist.angular.x
        q= odom_msg.twist.twist.angular.y
        r= odom_msg.twist.twist.angular.z

        #current_state = np.array([x,y,z,eta0,eps1,eps2,eps3,u,v,w,p,q,r]) #wth quaternions
        current_state = np.array([x,y,z,roll,pitch,yaw,u,v,w,p,q,r]) #wth euler angles

        return [current_state]

    def publish_control_actions(self):
        # Publish to relevant actuators here
        #od = Odometry()
        #print('Od:',od)
        #[self.current_x,self.current_y] = self.get_state_feedback(od)
        #yref= self.setpoint_cb(od)

        # Compute optimal control action, the feedback is updated automatically wth the feedback callback when new data is available
        #self.current_setpoint = self.dummy_setpoint()
        self.computeMPC()

        #Filter out when to publish the actuator commands
        tv_limit = 0.01
        rpm_limit= 50
        vbs_limit= 0.01
        lcg_limit = 0.01
        
        thruster1 = ThrusterRPM()
        thruster2 = ThrusterRPM()
        thruster1.rpm = self.prev_u[0]
        thruster2.rpm = self.prev_u[1]
        #thruster1.rpm = 500 # only controlling de,dr
        #thruster2.rpm = 500

        vec = ThrusterAngles()
        vec.thruster_horizontal_radians = self.prev_u[2]
        vec.thruster_vertical_radians = self.prev_u[3] 
        #vec.thruster_horizontal_radians = self.prev_u[0] # only controlling de, dr
        #vec.thruster_vertical_radians = self.prev_u[1] 

        vbs = PercentStamped()
        vbs.value = self.prev_u[4]
        
        lcg = PercentStamped()
        lcg.value = self.prev_u[5]

        if (self.counter % 5 == 0):
            #Check rpms
            if (abs(self.current_u[0]-self.prev_u[0])>rpm_limit) or (abs(self.current_u[1]-self.prev_u[1])>rpm_limit): 
                thruster1.rpm = self.current_u[0]
                thruster2.rpm = self.current_u[1]
                self.prev_u[0] = self.current_u[0]
                self.prev_u[1] = self.current_u[1]

            #check thrust vector
            #if(abs(self.current_u[0]-self.prev_u[0])>tv_limit) or (abs(self.current_u[1]-self.prev_u[1])>tv_limit): 
            if(abs(self.current_u[2]-self.prev_u[2])>tv_limit) or (abs(self.current_u[3]-self.prev_u[3])>tv_limit):                
                vec.thruster_horizontal_radians = self.current_u[2]
                vec.thruster_vertical_radians = self.current_u[3]   
                self.prev_u[2] = self.current_u[2]
                self.prev_u[3] = self.current_u[3]
                #vec.thruster_horizontal_radians = self.current_u[0] # only controlling de,dr
                #vec.thruster_vertical_radians = self.current_u[1]   
                #self.prev_u[0] = self.current_u[0]
                #self.prev_u[1] = self.current_u[1]
            
            #check vbs and lcg
            #if(abs(self.current_u[4]-self.prev_u[4])>vbs_limit): 
            vbs.value = (self.current_u[4]+1.0)*50.
            self.prev_u[4] = self.current_u[4]
            
            #if(abs(self.current_u[5]-self.prev_u[5])>lcg_limit): 
            lcg.value = (self.current_u[5]+1.0)*50.
            self.prev_u[5] = self.current_u[5]

            rospy.loginfo_throttle(5,'Publishing to actuators:')
            rospy.loginfo_throttle(5,self.prev_u)

            self.rpm1_pub.publish(thruster1)
            self.rpm2_pub.publish(thruster2)
            self.vec_pub.publish(vec)
            self.vbs_pub.publish(vbs)
            self.lcg_pub.publish(lcg)

    # Function to subscribe to reference trajectory
    def setpoint_cb(self,odom_msg):
        # Insert functon to get the publshed reference trajectory here
        #print('getting setpont')
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z
        eta0 = odom_msg.pose.pose.orientation.w
        eps1 = odom_msg.pose.pose.orientation.x
        eps2 = odom_msg.pose.pose.orientation.y
        eps3 = odom_msg.pose.pose.orientation.z

        #converting quaternion to euler
        rpy= euler_from_quaternion([eps1,eps2,eps3,eta0])
        roll = rpy[0]
        pitch = -rpy[1] #check sign
        yaw = -rpy[2] #check sign

        u = odom_msg.twist.twist.linear.x
        v = odom_msg.twist.twist.linear.y
        w = odom_msg.twist.twist.linear.z        
        p= odom_msg.twist.twist.angular.x
        q= odom_msg.twist.twist.angular.y
        r= odom_msg.twist.twist.angular.z
        self.current_setpoint = np.array([x,y,z,roll,pitch,yaw,u,v,w,p,q,r])

    def feedback_cb(self, odom_fb):
        #Feedback callback, runs MPC to compute the next control action each time it gets state feedback
        #print('computing control each time we recieve new feedback')
        self.current_x = self.get_state_feedback(odom_fb)
        #if self.counter % 10 == 0:
        #    print('feedback (13 elements)=', self.current_x)
    
    # Callback function to check for enable flag
    def enable_cb(self,enable_msg):
        #print('Enable:', enable_msg.data)
        if (not enable_msg.data):
            self.enable_flag = False 
            rospy.loginfo_throttle(5,'MPC disabled')

        else:
            self.enable_flag = True

    def __init__(self, name):
        # Constructor- instantiate parameters, publishers, subscribers
        self.xy_tolerance = rospy.get_param("~xy_tolerance", 5.0)
        self.depth_tolerance = rospy.get_param("~depth_tolerance", 0.5)
        self.prediction_horizon = rospy.get_param("~prediction_horizon", 5)
        self.loop_freq = rospy.get_param("~loop_freq", 11)

        # Topics for feedback and actuators
        state_feedback_topic = rospy.get_param("~state_feedback_topic", "/sam/sim/odom")
        setpoint_topic = rospy.get_param("~setpoint_topic", "/sam/ctrl/mpc/setpoint")
        rpm1_topic = rospy.get_param("~rpm_topic_1", "/sam/core/thruster1_cmd")
        rpm2_topic = rospy.get_param("~rpm_topic_1", "/sam/core/thruster2_cmd")
        vbs_topic = rospy.get_param("~vbs_topic", "/sam/core/vbs_cmd")
        lcg_topic = rospy.get_param("~lcg_topic", "/sam/core/lcg_cmd")
        thrust_vector_cmd_topic = rospy.get_param("~thrust_vector_cmd_topic", "/sam/core/thrust_vector_cmd")
        enable_topic = rospy.get_param("~enable_topic", "/sam/ctrl/mpc/enable")

        # Subscribers to state feedback, setpoints and enable flags
        rospy.Subscriber(state_feedback_topic, Odometry, self.feedback_cb)
        rospy.Subscriber(setpoint_topic, Odometry, self.setpoint_cb)
        rospy.Subscriber(enable_topic, Bool, self.enable_cb)

        # Publishers to actuators
        self.rpm1_pub = rospy.Publisher(rpm1_topic, ThrusterRPM, queue_size=10)
        self.rpm2_pub = rospy.Publisher(rpm2_topic, ThrusterRPM, queue_size=10)
        self.vec_pub = rospy.Publisher(thrust_vector_cmd_topic, ThrusterAngles, queue_size=10)
        self.vbs_pub = rospy.Publisher(vbs_topic, PercentStamped, queue_size=10)
        self.lcg_pub = rospy.Publisher(lcg_topic, PercentStamped, queue_size=10)

        rate = rospy.Rate(self.loop_freq) 
        
        #Initializing some 'global' variables
        self.enable_flag = True
        self.counter = 0
        self.epsi = 10e-5 #adding a tiny term to make the linearization numerically feasible and not singular or too sparse making the solver not converge
        self.current_x = np.array([-10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        #self.current_y = np.array([-10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.current_u = np.array([1., 1., 0.01, 0.01, 0.1, 0.1])
        self.prev_u = np.array([1, 1, 0.01, 0.01, 0.1, 0.1])
        #self.current_u = np.array([ 0.1, 0.1]) #controlling only de,dr
        #self.prev_u = np.array([ 0.1, 0.1])
        #self.current_u = np.array([ 1, 1, 0.1, 0.1]) #controlling only rpm1,rpm2,de,dr
        #self.prev_u = np.array([1, 1, 0.1, 0.1])
        #self.current_setpoint = np.array([20, 20, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.current_setpoint = np.array([15, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
        [self.A_lin, self.B_lin] = self.getLinearModel(self.current_x, self.current_u)
        
        while not rospy.is_shutdown():

            if(self.enable_flag):
                #self.test_mpc()
                goal_error = np.linalg.norm(self.current_x[:3]-self.current_setpoint[:3])

                if (goal_error > self.xy_tolerance):
                    rospy.loginfo_throttle(5,'Error norm = %f', goal_error)
                    self.publish_control_actions()
                #rospy.loginfo_throttle(1,'Running node')

                else:
                    print('Goal reached, holding position')
            
            rate.sleep()
        #rospy.spin() # in case I want to be event driven, like a subscriber


if __name__ == "__main__":
    rospy.init_node("mpc_controller")
    controller = MPC_Controller(rospy.get_name())
    
