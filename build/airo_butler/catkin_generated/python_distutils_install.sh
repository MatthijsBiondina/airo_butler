#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/matt/catkin_ws/src/airo_butler"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/matt/catkin_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/matt/catkin_ws/install/lib/python3/dist-packages:/home/matt/catkin_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/matt/catkin_ws/build" \
    "/home/matt/anaconda3/envs/airo-mono/bin/python3" \
    "/home/matt/catkin_ws/src/airo_butler/setup.py" \
     \
    build --build-base "/home/matt/catkin_ws/build/airo_butler" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/matt/catkin_ws/install" --install-scripts="/home/matt/catkin_ws/install/bin"
