;; Auto-generated. Do not edit!


(when (boundp 'airo_butler::MoveToJointConfiguration)
  (if (not (find-package "AIRO_BUTLER"))
    (make-package "AIRO_BUTLER"))
  (shadow 'MoveToJointConfiguration (find-package "AIRO_BUTLER")))
(unless (find-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATION")
  (make-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATION"))
(unless (find-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATIONREQUEST")
  (make-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATIONREQUEST"))
(unless (find-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATIONRESPONSE")
  (make-package "AIRO_BUTLER::MOVETOJOINTCONFIGURATIONRESPONSE"))

(in-package "ROS")





(defclass airo_butler::MoveToJointConfigurationRequest
  :super ros::object
  :slots (_pod ))

(defmethod airo_butler::MoveToJointConfigurationRequest
  (:init
   (&key
    ((:pod __pod) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _pod __pod)
   self)
  (:pod
   (&optional __pod)
   (if __pod (setq _pod __pod)) _pod)
  (:serialization-length
   ()
   (+
    ;; uint8[] _pod
    (* 1    (length _pod)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8[] _pod
     (write-long (length _pod) s)
     (princ _pod s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8[] _pod
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _pod (make-array n :element-type :char))
     (replace _pod buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(defclass airo_butler::MoveToJointConfigurationResponse
  :super ros::object
  :slots (_success ))

(defmethod airo_butler::MoveToJointConfigurationResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass airo_butler::MoveToJointConfiguration
  :super ros::object
  :slots ())

(setf (get airo_butler::MoveToJointConfiguration :md5sum-) "fd3e7601c92afe400a111ea24ac875e1")
(setf (get airo_butler::MoveToJointConfiguration :datatype-) "airo_butler/MoveToJointConfiguration")
(setf (get airo_butler::MoveToJointConfiguration :request) airo_butler::MoveToJointConfigurationRequest)
(setf (get airo_butler::MoveToJointConfiguration :response) airo_butler::MoveToJointConfigurationResponse)

(defmethod airo_butler::MoveToJointConfigurationRequest
  (:response () (instance airo_butler::MoveToJointConfigurationResponse :init)))

(setf (get airo_butler::MoveToJointConfigurationRequest :md5sum-) "fd3e7601c92afe400a111ea24ac875e1")
(setf (get airo_butler::MoveToJointConfigurationRequest :datatype-) "airo_butler/MoveToJointConfigurationRequest")
(setf (get airo_butler::MoveToJointConfigurationRequest :definition-)
      "uint8[] pod
---
bool success
")

(setf (get airo_butler::MoveToJointConfigurationResponse :md5sum-) "fd3e7601c92afe400a111ea24ac875e1")
(setf (get airo_butler::MoveToJointConfigurationResponse :datatype-) "airo_butler/MoveToJointConfigurationResponse")
(setf (get airo_butler::MoveToJointConfigurationResponse :definition-)
      "uint8[] pod
---
bool success
")



(provide :airo_butler/MoveToJointConfiguration "fd3e7601c92afe400a111ea24ac875e1")


