
(cl:in-package :asdf)

(defsystem "airo_butler-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "PODService" :depends-on ("_package_PODService"))
    (:file "_package_PODService" :depends-on ("_package"))
  ))