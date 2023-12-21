execute_process(COMMAND "/home/matt/catkin_ws/build/airo_butler/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/matt/catkin_ws/build/airo_butler/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
