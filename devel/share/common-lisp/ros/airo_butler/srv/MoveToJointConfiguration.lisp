; Auto-generated. Do not edit!


(cl:in-package airo_butler-srv)


;//! \htmlinclude MoveToJointConfiguration-request.msg.html

(cl:defclass <MoveToJointConfiguration-request> (roslisp-msg-protocol:ros-message)
  ((pod
    :reader pod
    :initarg :pod
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass MoveToJointConfiguration-request (<MoveToJointConfiguration-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MoveToJointConfiguration-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MoveToJointConfiguration-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name airo_butler-srv:<MoveToJointConfiguration-request> is deprecated: use airo_butler-srv:MoveToJointConfiguration-request instead.")))

(cl:ensure-generic-function 'pod-val :lambda-list '(m))
(cl:defmethod pod-val ((m <MoveToJointConfiguration-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader airo_butler-srv:pod-val is deprecated.  Use airo_butler-srv:pod instead.")
  (pod m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MoveToJointConfiguration-request>) ostream)
  "Serializes a message object of type '<MoveToJointConfiguration-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'pod))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'pod))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MoveToJointConfiguration-request>) istream)
  "Deserializes a message object of type '<MoveToJointConfiguration-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'pod) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'pod)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MoveToJointConfiguration-request>)))
  "Returns string type for a service object of type '<MoveToJointConfiguration-request>"
  "airo_butler/MoveToJointConfigurationRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveToJointConfiguration-request)))
  "Returns string type for a service object of type 'MoveToJointConfiguration-request"
  "airo_butler/MoveToJointConfigurationRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MoveToJointConfiguration-request>)))
  "Returns md5sum for a message object of type '<MoveToJointConfiguration-request>"
  "fd3e7601c92afe400a111ea24ac875e1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MoveToJointConfiguration-request)))
  "Returns md5sum for a message object of type 'MoveToJointConfiguration-request"
  "fd3e7601c92afe400a111ea24ac875e1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MoveToJointConfiguration-request>)))
  "Returns full string definition for message of type '<MoveToJointConfiguration-request>"
  (cl:format cl:nil "uint8[] pod~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MoveToJointConfiguration-request)))
  "Returns full string definition for message of type 'MoveToJointConfiguration-request"
  (cl:format cl:nil "uint8[] pod~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MoveToJointConfiguration-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'pod) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MoveToJointConfiguration-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MoveToJointConfiguration-request
    (cl:cons ':pod (pod msg))
))
;//! \htmlinclude MoveToJointConfiguration-response.msg.html

(cl:defclass <MoveToJointConfiguration-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass MoveToJointConfiguration-response (<MoveToJointConfiguration-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MoveToJointConfiguration-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MoveToJointConfiguration-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name airo_butler-srv:<MoveToJointConfiguration-response> is deprecated: use airo_butler-srv:MoveToJointConfiguration-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <MoveToJointConfiguration-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader airo_butler-srv:success-val is deprecated.  Use airo_butler-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MoveToJointConfiguration-response>) ostream)
  "Serializes a message object of type '<MoveToJointConfiguration-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MoveToJointConfiguration-response>) istream)
  "Deserializes a message object of type '<MoveToJointConfiguration-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MoveToJointConfiguration-response>)))
  "Returns string type for a service object of type '<MoveToJointConfiguration-response>"
  "airo_butler/MoveToJointConfigurationResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveToJointConfiguration-response)))
  "Returns string type for a service object of type 'MoveToJointConfiguration-response"
  "airo_butler/MoveToJointConfigurationResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MoveToJointConfiguration-response>)))
  "Returns md5sum for a message object of type '<MoveToJointConfiguration-response>"
  "fd3e7601c92afe400a111ea24ac875e1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MoveToJointConfiguration-response)))
  "Returns md5sum for a message object of type 'MoveToJointConfiguration-response"
  "fd3e7601c92afe400a111ea24ac875e1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MoveToJointConfiguration-response>)))
  "Returns full string definition for message of type '<MoveToJointConfiguration-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MoveToJointConfiguration-response)))
  "Returns full string definition for message of type 'MoveToJointConfiguration-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MoveToJointConfiguration-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MoveToJointConfiguration-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MoveToJointConfiguration-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MoveToJointConfiguration)))
  'MoveToJointConfiguration-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MoveToJointConfiguration)))
  'MoveToJointConfiguration-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveToJointConfiguration)))
  "Returns string type for a service object of type '<MoveToJointConfiguration>"
  "airo_butler/MoveToJointConfiguration")