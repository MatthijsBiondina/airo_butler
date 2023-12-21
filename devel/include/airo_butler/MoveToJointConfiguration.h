// Generated by gencpp from file airo_butler/MoveToJointConfiguration.msg
// DO NOT EDIT!


#ifndef AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATION_H
#define AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATION_H

#include <ros/service_traits.h>


#include <airo_butler/MoveToJointConfigurationRequest.h>
#include <airo_butler/MoveToJointConfigurationResponse.h>


namespace airo_butler
{

struct MoveToJointConfiguration
{

typedef MoveToJointConfigurationRequest Request;
typedef MoveToJointConfigurationResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct MoveToJointConfiguration
} // namespace airo_butler


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::airo_butler::MoveToJointConfiguration > {
  static const char* value()
  {
    return "fd3e7601c92afe400a111ea24ac875e1";
  }

  static const char* value(const ::airo_butler::MoveToJointConfiguration&) { return value(); }
};

template<>
struct DataType< ::airo_butler::MoveToJointConfiguration > {
  static const char* value()
  {
    return "airo_butler/MoveToJointConfiguration";
  }

  static const char* value(const ::airo_butler::MoveToJointConfiguration&) { return value(); }
};


// service_traits::MD5Sum< ::airo_butler::MoveToJointConfigurationRequest> should match
// service_traits::MD5Sum< ::airo_butler::MoveToJointConfiguration >
template<>
struct MD5Sum< ::airo_butler::MoveToJointConfigurationRequest>
{
  static const char* value()
  {
    return MD5Sum< ::airo_butler::MoveToJointConfiguration >::value();
  }
  static const char* value(const ::airo_butler::MoveToJointConfigurationRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::airo_butler::MoveToJointConfigurationRequest> should match
// service_traits::DataType< ::airo_butler::MoveToJointConfiguration >
template<>
struct DataType< ::airo_butler::MoveToJointConfigurationRequest>
{
  static const char* value()
  {
    return DataType< ::airo_butler::MoveToJointConfiguration >::value();
  }
  static const char* value(const ::airo_butler::MoveToJointConfigurationRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::airo_butler::MoveToJointConfigurationResponse> should match
// service_traits::MD5Sum< ::airo_butler::MoveToJointConfiguration >
template<>
struct MD5Sum< ::airo_butler::MoveToJointConfigurationResponse>
{
  static const char* value()
  {
    return MD5Sum< ::airo_butler::MoveToJointConfiguration >::value();
  }
  static const char* value(const ::airo_butler::MoveToJointConfigurationResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::airo_butler::MoveToJointConfigurationResponse> should match
// service_traits::DataType< ::airo_butler::MoveToJointConfiguration >
template<>
struct DataType< ::airo_butler::MoveToJointConfigurationResponse>
{
  static const char* value()
  {
    return DataType< ::airo_butler::MoveToJointConfiguration >::value();
  }
  static const char* value(const ::airo_butler::MoveToJointConfigurationResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // AIRO_BUTLER_MESSAGE_MOVETOJOINTCONFIGURATION_H
