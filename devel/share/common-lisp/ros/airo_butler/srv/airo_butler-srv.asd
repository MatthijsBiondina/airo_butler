
(cl:in-package :asdf)

(defsystem "airo_butler-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "MoveToJointConfiguration" :depends-on ("_package_MoveToJointConfiguration"))
    (:file "_package_MoveToJointConfiguration" :depends-on ("_package"))
  ))