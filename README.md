Informal Documentation of the project code:

## Launching the Code

- Project code is located at `/python/examples/project_nav_man`, but requires & expects a ROS Noetic environment with the Spot ROS1 driver installed. As ROS Noetic has been discontinued, any project code needs to be ported to ROS2 for further use & development.
- For the sake of testing the existing project code I am building a Dockerfile that uses the `osrf/ros:noetic-desktop-full` image as its base since ROS Noetic can no longer be installed through `apt`


Original README begins below.

---

<!--
Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.

Downloading, reproducing, distributing or otherwise using the SDK Software
is subject to the terms and conditions of the Boston Dynamics Software
Development Kit License (20191101-BDSDK-SL).
-->

<p class="github-only">
<b>The Spot SDK documentation is best viewed via our developer site at <a href="https://dev.bostondynamics.com">dev.bostondynamics.com</a>. </b>
</p>

# Spot SDK

Develop applications and payloads for Spot using the Boston Dynamics Spot SDK.

The SDK consists of:
*  [Conceptual documentation](docs/concepts/README.md). These documents explain the key abstractions used by the Spot API.
*  [Python client library](docs/python/README.md). Applications using the Python library can control Spot and read sensor and health information from Spot. A wide variety of example programs and a QuickStart guide are also included.
*  [Payload developer documentation](docs/payload/README.md). Payloads add additional sensing, communication, and control capabilities beyond what the base platform provides. The Payload ICD covers the mechanical, electrical, and software interfaces that Spot supports.
*  [Spot API protocol definition](protos/bosdyn/api/README.md). This reference guide covers the details of the protocol applications used to communicate to Spot. Application developers who wish to use a language other than Python can implement clients that speak the protocol.
*  [Spot SDK Repository](https://github.com/boston-dynamics/spot-sdk). The GitHub repo where all of the Spot SDK code is hosted.

This is version 3.2.2.post1 of the SDK. Please review the [Release Notes](docs/release_notes.md) to see what has changed.

## Contents

* [Concepts](docs/concepts/README.md)
* [Python](docs/python/README.md)
* [Payloads](docs/payload/README.md)
* [API Protocol](docs/protos/README.md)
* [Release Notes](docs/release_notes.md)
* [SDK Repository](https://github.com/boston-dynamics/spot-sdk)
* [Scout](docs/scout/index.md)
