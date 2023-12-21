// Generated by gencpp from file airo_butler/MoveToJointConfigurationRequest.msg
// DO NOT EDIT!


#ifndef AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATIONREQUEST_H
#define AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATIONREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace airo_butler
{
template <class ContainerAllocator>
struct MoveToJointConfigurationRequest_
{
  typedef MoveToJointConfigurationRequest_<ContainerAllocator> Type;

  MoveToJointConfigurationRequest_()
    : pod()  {
    }
  MoveToJointConfigurationRequest_(const ContainerAllocator& _alloc)
    : pod(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>> _pod_type;
  _pod_type pod;





  typedef boost::shared_ptr< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> const> ConstPtr;

}; // struct MoveToJointConfigurationRequest_

typedef ::airo_butler::MoveToJointConfigurationRequest_<std::allocator<void> > MoveToJointConfigurationRequest;

typedef boost::shared_ptr< ::airo_butler::MoveToJointConfigurationRequest > MoveToJointConfigurationRequestPtr;
typedef boost::shared_ptr< ::airo_butler::MoveToJointConfigurationRequest const> MoveToJointConfigurationRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator1> & lhs, const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator2> & rhs)
{
  return lhs.pod == rhs.pod;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator1> & lhs, const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace airo_butler

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5d9014aa0b47c5915dec913295ebd501";
  }

  static const char* value(const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5d9014aa0b47c591ULL;
  static const uint64_t static_value2 = 0x5dec913295ebd501ULL;
};

template<class ContainerAllocator>
struct DataType< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "airo_butler/MoveToJointConfigurationRequest";
  }

  static const char* value(const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8[] pod\n"
;
  }

  static const char* value(const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.pod);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct MoveToJointConfigurationRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::airo_butler::MoveToJointConfigurationRequest_<ContainerAllocator>& v)
  {
    s << indent << "pod[]" << std::endl;
    for (size_t i = 0; i < v.pod.size(); ++i)
    {
      s << indent << "  pod[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.pod[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATIONREQUEST_H