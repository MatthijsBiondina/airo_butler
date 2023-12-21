// Auto-generated. Do not edit!

// (in-package airo_butler.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class MoveToJointConfigurationRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pod = null;
    }
    else {
      if (initObj.hasOwnProperty('pod')) {
        this.pod = initObj.pod
      }
      else {
        this.pod = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MoveToJointConfigurationRequest
    // Serialize message field [pod]
    bufferOffset = _arraySerializer.uint8(obj.pod, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MoveToJointConfigurationRequest
    let len;
    let data = new MoveToJointConfigurationRequest(null);
    // Deserialize message field [pod]
    data.pod = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.pod.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'airo_butler/MoveToJointConfigurationRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5d9014aa0b47c5915dec913295ebd501';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8[] pod
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MoveToJointConfigurationRequest(null);
    if (msg.pod !== undefined) {
      resolved.pod = msg.pod;
    }
    else {
      resolved.pod = []
    }

    return resolved;
    }
};

class MoveToJointConfigurationResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MoveToJointConfigurationResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MoveToJointConfigurationResponse
    let len;
    let data = new MoveToJointConfigurationResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'airo_butler/MoveToJointConfigurationResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MoveToJointConfigurationResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: MoveToJointConfigurationRequest,
  Response: MoveToJointConfigurationResponse,
  md5sum() { return 'fd3e7601c92afe400a111ea24ac875e1'; },
  datatype() { return 'airo_butler/MoveToJointConfiguration'; }
};
