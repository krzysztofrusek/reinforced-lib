.. _examples_page:

########
Examples
########

.. _gym_integration:

**************************
Integration with Gymnasium
**************************

`Gymnasium <https://gymnasium.farama.org/>`_, formerly known as OpenAI Gym, is a popular toolkit that provides a standardized interface for reinforcement learning environments. Gymnasium offers a variety of environments, from simple classic control tasks like balancing a pole, which is described below in detail, to complex games like Atari and MuJoCo. It even supports creating custom environments, making it a versatile tool for all things reinforcement learning research.

Reinforced-lib on the other hand provides implementations of various reinforcement learning algorithms. It can seamlessly integrate with Gymnasium by using the environments provided by Gymnasium as the learning context for the algorithms implemented in Reinforced-lib. This integration allows developers to easily train and evaluate their reinforcement learning models using a wide variety of pre-defined scenarios.

Cart Pole example
=================

The Cart Pole environment is a classic control task in which the goal is to balance a pole on a cart. The environment is described by a 4-dimensional state space, which consists of the cart's position, the cart's velocity, the pole's angle, and the pole's angular velocity. The agent can take one of two actions: push the cart to the left or push the cart to the right. The episode ends when the pole falls below a certain angle or the cart moves outside of the environment's boundaries. The goal is to keep the pole balanced for as long as possible.

The following example demonstrates how to train a reinforcement learning agent using Reinforced-lib and Gymnasium. The agent uses the :ref:`Deep Q-Learning (DQN) <Deep Q-Learning (DQN)>` algorithm to learn how to balance the pole. The DQN algorithm is implemented in Reinforced-lib and the Cart Pole environment is provided by Gymnasium.

We start with the necessary imports:

.. code-block:: python

    import gymnasium as gym
    import optax
    from chex import Array
    from flax import linen as nn

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import DQN
    from reinforced_lib.exts import Gymnasium
    from reinforced_lib.logs import StdoutLogger, TensorboardLogger

We than define the QNetwork approximator as a simple multi-layer perceptron with a ReLU activation function:

.. code-block:: python

    class QNetwork(nn.Module):
        @nn.compact
        def __call__(self, x: Array) -> Array:
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            return nn.Dense(2)(x)

Next, we step into the ``run`` function, which is responsible for training the agent. We start by instantiating the Reinforced-lib, specifying the agent as a DQN, the extension as :ref:`Gymnasium <Gymnasium>`, and the loggers as :ref:`StdoutLogger <StdoutLogger>` and :ref:`TensorboardLogger <TensorboardLogger>` to log both to the console and to a TensorBoard file.  Note that we specify the environment type in the parameters of the extension to allow for automatic inference of environment properties, such as the state and action space sizes.

.. code-block:: python

    def run(num_epochs: int) -> None:
        rl = RLib(
            agent_type=DQN,
            agent_params={
                'q_network': QNetwork(),
                'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2),
                'discount': 0.95,
                'epsilon_decay': 0.9975
            },
            ext_type=Gymnasium,
            ext_params={'env_id': 'CartPole-v1'},
            logger_types=[StdoutLogger, TensorboardLogger]
        )

We then start the training loop where we iterate over the number of epochs and for each epoch we let the agent interact with the environment. We start by resetting the environment and sampling the initial action of the agent. Then we run the agent in the environment by performing the action in the environment and sampling the next action. We continue this loop until the environment reaches a terminal state. We log the length of the epoch as the performance metric and move on to the next epoch.

.. code-block:: python

        for epoch in range(num_epochs):
            env = gym.make('CartPole-v1', render_mode='no')

            _, _ = env.reset()
            action = env.action_space.sample()
            terminal = False
            epoch_len = 0

            while not terminal:
                env_state = env.step(action.item())
                action = rl.sample(*env_state)
                terminal = env_state[2] or env_state[3]
                epoch_len += 1
            
            rl.log('epoch_len', epoch_len)

We start the training by calling the ``run`` function with the number of epochs as an argument:

.. code-block:: python

    if __name__ == '__main__':
        run(num_epochs=300)

The complete, runnable code can be copy pasted from the following snippet:

.. code-block:: python

    import gymnasium as gym
    import optax
    from chex import Array
    from flax import linen as nn

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import DQN
    from reinforced_lib.exts import Gymnasium
    from reinforced_lib.logs import StdoutLogger, TensorboardLogger


    class QNetwork(nn.Module):
        @nn.compact
        def __call__(self, x: Array) -> Array:
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            return nn.Dense(2)(x)


    def run(num_epochs: int) -> None:
        rl = RLib(
            agent_type=DQN,
            agent_params={
                'q_network': QNetwork(),
                'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2),
                'discount': 0.95,
                'epsilon_decay': 0.9975
            },
            ext_type=Gymnasium,
            ext_params={'env_id': 'CartPole-v1'},
            logger_types=[StdoutLogger, TensorboardLogger]
        )

        for epoch in range(num_epochs):
            env = gym.make('CartPole-v1', render_mode='no')

            _, _ = env.reset()
            action = env.action_space.sample()
            terminal = False
            epoch_len = 0

            while not terminal:
                env_state = env.step(action.item())
                action = rl.sample(*env_state)
                terminal = env_state[2] or env_state[3]
                epoch_len += 1
            
            rl.log('epoch_len', epoch_len)


    if __name__ == '__main__':
        run(num_epochs=300)

Other examples
==============

We provide a few more examples of Reinforced-lib and Gymnasium integration in the `examples <https://github.com/m-wojnar/reinforced-lib/tree/main/examples>`_ directory of the Reinforced-lib repository. The examples include the training of the DQN agent in the `Cart Pole environment <https://github.com/m-wojnar/reinforced-lib/tree/main/examples/cart-pole>`_ (described above) and the training of the DDPG agent in the `Pendulum environment <https://github.com/m-wojnar/reinforced-lib/tree/main/examples/pendulum>`_. The examples are fully runnable and can be used as a starting point for your own reinforcement learning experiments with Reinforced-lib and Gymnasium.


.. _ns3_connection:

********************
Connection with ns-3
********************

We will demonstrate the cooperation of Reinforced-lib with an external Wi-Fi simulation software based on an example of
an ML-controlled rate adaptation (RA) manager. To simulate the Wi-Fi environment, we will use the popular, research
oriented network simulator -- ns-3. To learn more about the simulator, we encourage to visit the
`official website <https://www.nsnam.org/>`_ or read the
`ns-3 tutorial <https://www.nsnam.org/docs/release/3.36/tutorial/html/index.html>`_.


Docker container setup
======================

To facilitate the setup of the Reinforced-lib and ns-3 connection, we provide a Dockerfile that contains all the necessary
dependencies and configurations. You need to have Docker installed on your machine, which you can download from the
`Docker website <https://www.docker.com/get-started>`_.

To build the Docker image, use the Dockerfile `provided in the repository <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/Dockerfile>`_.
Navigate to the directory where the Dockerfile is located and run the following command ("rlib-ns3" is the name of the image):

.. code-block:: bash

    docker build -t "rlib-ns3" .

Once the image is built, you can run the interactive session with the following command:

.. code-block:: bash

    docker run -it "rlib-ns3" bash

To persist the changes made in the container, you can create a volume and mount it to the container by adding the ``-v``
flag to the ``docker run`` command:

.. code-block:: bash

    docker volume create "rlib-ns3-data"
    docker run -it -v "rlib-ns3-data:/home" "rlib-ns3" bash

Reinforced-lib and ns-3 are already installed in the container, so you can proceed with the experiments described in the
:ref:`simulation scenario section <Simulation scenario>`. The library is located in the ``/home/reinforced-lib`` directory and the
ns-3 in the ``/home/ns-3-dev`` directory.


Manual setup
============

To perform experiments with Python-based Reinforced-lib and C++-based ns-3, you need to setup an environment which
consists of the following:

  * favourite C++ compiler (we assume that you already have one in your dev stack),
  * ns-3 (connection tested on the ns-3.37 version),
  * ns3-ai (`GitHub repository <https://github.com/hust-diangroup/ns3-ai/>`_).

Since the ns-3 requires the compilation, we will install all the required modules, transfer ns-3 files required for the
communication with Reinforced-lib, and compile everything once at the very end.


Installing ns-3
---------------

There are a few ways to install ns-3, all described in the `ns-3 wiki <https://www.nsnam.org/wiki/Installation>`_,
but we recommend to install ns-3 by cloning the git dev repository:

.. code-block:: bash

    git clone https://gitlab.com/nsnam/ns-3-dev.git

We recommend setting the simulator to the 3.37 version, since we do not guarantee the compatibility with other versions.
To set the ns-3 to the 3.37:

.. code-block:: bash

    cd ns-3-dev     # this directory will be referenced as YOUR_NS3_PATH since now on
    git reset --hard 4407a9528eac81476546a50597cc6e016a428f43


Installing ns3-ai
-----------------

The ns3-ai module interconnects ns-3 and Reinforced-lib (or any other python-writen software) by transferring data through
the shared memory pool. The memory is accessed by both sides thus making the connection. You can read more about the ns3-ai on the
`ns3-ai official repository <https://github.com/hust-diangroup/ns3-ai>`_.

.. note::

    ns3-ai (as of 10.08.2024) is aligned with the latest versions of ns-3. We recommend resetting the repository to a specific version to make it compatible with version 3.37.

.. code-block:: bash

    cd $YOUR_NS3_PATH/contrib/
    git clone https://github.com/hust-diangroup/ns3-ai.git
    cd ns3-ai
    git reset --hard 86453e840c6e5df849d8c4e9c7f88eade637798c
    pip install "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"


Transferring ns3 files
----------------------

In ``$REINFORCED_LIB/examples/ns-3-ra/`` there are two directories. The ``scratch`` contains an
example RA scenario, which will be described in the :ref:`next section <rlib-sim>`. The ``contrib`` directory
contains a ``rlib-wifi-manager`` module with the specification of a custom rate adaptation manager that communicates with python
with the use of ns3-ai. You need to transfer both of these directories in the appropriate locations by running the
following commands:

.. code-block:: bash

    cp $REINFORCED_LIB/examples/ns-3-ra/scratch/* $YOUR_NS3_PATH/scratch/
    cp -r $REINFORCED_LIB/examples/ns-3-ra/contrib/rlib-wifi-manager $YOUR_NS3_PATH/contrib/

.. note::

    To learn more about adding contrib modules to ns-3, visit
    the `ns-3 manual <https://www.nsnam.org/docs/manual/html/new-modules.html>`_.


Compiling ns3
-------------

To have the simulator working and fully integrated with the Reinforced-lib, we need to compile it. We do this from the ``YOUR_NS3_PATH`` in two steps, by first configuring the compilation and then by building ns-3:

.. code-block:: bash

    cd $YOUR_NS3_PATH
    ./ns3 configure --build-profile=optimized --disable-examples --disable-tests
    ./ns3 build

Once you have built ns-3, you can test the ns-3 and Reinforced-lib integration by executing the script that runs an example
rate adaptation scenario controlled by the UCB agent.

.. code-block:: bash

    cd $REINFORCED_LIB
    ./test/test_ns3_integration.sh $YOUR_NS3_PATH

On success, in your home directory, there should be a ``rlib-ns3-integration-test.csv`` file generated filled with some data.

.. _rlib-sim:


Simulation scenario
===================

ns-3 (C++) part
---------------

In ``rscratch`` directory we supply an example scenario ``rlib-sim.cc`` to test the rate adaptation manager in the 802.11ax
environment. The scenario is highly customizable but the key points
are that there is one access point (AP) and a variable number (``--nWifi``) of stations (STA); there is an uplink, saturated
communication (from stations to AP) and the AP is in line of sight with all the stations; all the stations are at the point of
:math:`(0, 0)~m` and the AP can either be at :math:`(0, 0)~m` as well or in some distance (``--initialPosition``)
from the stations. The AP can also be moving with a constant velocity (``--velocity``) to simulate dynamic scenarios.
Other assumptions from the simulation are the A-MPDU frame aggregation, 5 Ghz frequency band, and single spatial stream.

By typing ``$YOUR_NS3_PATH/build/scratch/ns3.37-ra-sim-optimized --help`` you can go over the simulation parameters and
learn what is the function of each.

.. code-block:: bash

    ./build/scratch/ns3.37-ra-sim-optimized --help
    [Program Options] [General Arguments]

    Program Options:
        --area:             Size of the square in which stations are wandering (m) [RWPM mobility type] [40]
        --channelWidth:     Channel width (MHz) [20]
        --csvPath:          Save an output file in the CSV format
        --dataRate:         Aggregate traffic generators data rate (Mb/s) [125]
        --deltaPower:       Power change (dBm) [0]
        --initialPosition:  Initial position of the AP on X axis (m) [Distance mobility type] [0]
        --intervalPower:    Interval between power change (s) [4]
        --logEvery:         Time interval between successive measurements (s) [1]
        --lossModel:        Propagation loss model to use [LogDistance, Nakagami] [LogDistance]
        --minGI:            Shortest guard interval (ns) [3200]
        --mobilityModel:    Mobility model [Distance, RWPM] [Distance]
        --nodeSpeed:        Maximum station speed (m/s) [RWPM mobility type] [1.4]
        --nodePause:        Maximum time station waits in newly selected position (s) [RWPM mobility type] [20]
        --nWifi:            Number of transmitting stations [1]
        --pcapPath:         Save a PCAP file from the AP
        --simulationTime:   Duration of the simulation excluding warmup stage (s) [20]
        --velocity:         Velocity of the AP on X axis (m/s) [Distance mobility type] [0]
        --warmupTime:       Duration of the warmup stage (s) [2]
        --wifiManager:      Rate adaptation manager [ns3::RLibWifiManager]
        --wifiManagerName:  Name of the Wi-Fi manager in CSV

    General Arguments:
        --PrintGlobals:              Print the list of globals.
        --PrintGroups:               Print the list of groups.
        --PrintGroup=[group]:        Print all TypeIds of group.
        --PrintTypeIds:              Print all TypeIds.
        --PrintAttributes=[typeid]:  Print all attributes of typeid.
        --PrintVersion:              Print the ns-3 version.
        --PrintHelp:                 Print this help message.


Reinforced-lib (Python) part
----------------------------

The provided rate adaptation manager is implemented in the file ``$REINFORCED_LIB/examples/ns-3-ra/main.py``. Here we specify the
communication with the ns-3 simulator by defining the environment's observation space and the action space, we create the ``RLib``
agent, we provide the agent-environment interaction loop which reacts to the incoming (aggregated) frames by responding with an appropriate MCS,
and cleans up the environment when the simulation is done. Below we include and explain the essential fragments from the ``main.py`` script.

.. code-block:: python
    :linenos:
    :lineno-start: 4

    from ext import IEEE_802_11_ax_RA
    from particle_filter import ParticleFilter
    from py_interface import *   # Import the ns3-ai structures

    from reinforced_lib import RLib
    from reinforced_lib.agents.mab import *

We import the RA extension, agents and the RLib module. Line 6 is responsible for importing the structures from the ns3-ai
library.

.. code-block:: python
    :linenos:
    :lineno-start: 12

    class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('power', c_double),
        ('time', c_double),
        ('cw', c_uint32),
        ('n_failed', c_uint32),
        ('n_successful', c_uint32),
        ('n_wifi', c_uint32),
        ('station_id', c_uint32),
        ('type', c_uint8)
    ]


    class Act(Structure):
        _pack_ = 1
        _fields_ = [
            ('station_id', c_uint32),
            ('mcs', c_uint8)
        ]

Next we define the ns3-ai structures that describe the environment space and action space accordingly. The structures must
strictly reflect the ones defined in the 
`header file <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/contrib/rlib-wifi-manager/model/rlib-wifi-manager.h>`_
``contrib/rlib-wifi-manager/model/rlib-wifi-manager.h`` because it is the interface of the shared memory data bridge between
python and C++. You can learn more about the data exchange model
`here <https://github.com/hust-diangroup/ns3-ai/tree/master/examples/a_plus_b>`_.


.. code-block:: python
    :linenos:
    :lineno-start: 73

    rl = RLib(
        agent_type=agent_type,
        agent_params=agent_params,
        ext_type=IEEE_802_11_ax_RA
    )

    exp = Experiment(mempool_key, memory_size, 'ra-sim', ns3_path, using_waf=False)
    var = Ns3AIRL(memblock_key, Env, Act)

In line 73, we create an instance of RLib by supplying the appropriate, parametrized agent and the 802.11ax environment extension.
We define the ns3-ai experiment in line 79 by setting the memory key, the memory size, the name of the ns-3 scenario, and the path
to the ns3 root directory. In line 80, we create a handler to the shared memory interface by providing an arbitrary key and
the previously defined environment and action structures.


.. code-block:: python
    :linenos:
    :lineno-start: 82

    try:
        ns3_process = exp.run(ns3_args, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                if data.env.type == 0:
                    data.act.station_id = rl.init(seed)

                elif data.env.type == 1:
                    observation = {
                        'time': data.env.time,
                        'n_successful': data.env.n_successful,
                        'n_failed': data.env.n_failed,
                        'n_wifi': data.env.n_wifi,
                        'power': data.env.power,
                        'cw': data.env.cw
                    }

                    data.act.station_id = data.env.station_id
                    data.act.mcs = rl.sample(agent_id=data.env.station_id, **observation)

        ns3_process.wait()
    finally:
        del exp

The final step to make the example work is to define the agent-environment interaction loop. We loop while the ns3 simulation is running (line 85)
and if there is any data to be read (line 86). We differentiate the environment observation by a type attribute which
indicates whether it is an initialization frame or not. On initialization (line 90), we have to initialize our RL agent with
some seed. In the opposite case we translate the observation to a dictionary (lines 94-102) and override the action structure
with the received station ID (line 104) and the appropriate MCS selected by the RL agent (line 105). The last thing is to
clean up the shared memory environment when the simulation is finished (lines 107 and 107).


Example experiments
===================

We supply the ``$REINFORCED_LIB/examples/ns-3-ra/main.py`` script with the CLI so that you can test the rate adaptation manager in different
scenarios. We reflect all the command line arguments listed in :ref:`ns3 scenario <rlib-sim>` ``scratch/ra-sim.cc`` with four additional arguments:

  * ``--agent`` -- the type of RL agent responsible for the RA, a required argument,
  * ``--mempoolKey`` -- shared memory pool key, which is an arbitrary integer, greater than 1000, default is 1234.
  * ``--ns3Path`` -- path to the ns3 root directory, a required argument,

You can try running the following commands to test the Reinforced-lib rate adaptation manager in different example scenarios:

  a. Static scenario with 1 AP and 1 STA both positioned in the same place, RA handled by the *UCB* agent

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="UCB" --ns3Path="$YOUR_NS3_PATH"

  b. Static scenario with 1 AP and 1 STA both positioned in the same place, RA handled by the *UCB* agent. Output
  saved to the ``$HOME/ra-results.csv`` file and ``.pcap`` saved to the ``$HOME/ra-experiment-0-0.pcap``.

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="UCB" --ns3Path="$YOUR_NS3_PATH" --csvPath="$HOME/ra-results.scv" --pcapPath="$HOME/ra-experiment"

  c. Static scenario with 1 AP and 16 stations at a 10 m distance, RA handled by the *ThompsonSampling* agent.

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="ThompsonSampling" --ns3_path="$YOUR_NS3_PATH" --nWifi=16 --initialPosition=10

  d. Dynamic scenario with 1 AP and 1 STA starting at 0 m and moving away from AP with a velocity of 1 m/s, RA handled by the *ParticleFilter* agent.

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="ParticleFilter" --ns3Path="$YOUR_NS3_PATH" --velocity=1

Source code of the example
===========================

The complete, runnable code can be found in the `examples/ns-3-ra <https://github.com/m-wojnar/reinforced-lib/tree/main/examples/ns-3-ra>`_ directory of the Reinforced-lib repository. The example provides many useful scripts for reproducing our experiments and can be used as a starting point for your own reinforcement learning experiments with Reinforced-lib and ns-3. We also encourage you to see another example - implementation of the `centralized contention window optimization with DRL (CCOD) <https://ieeexplore.ieee.org/document/9417575?denied=>`_ in the ``examples/ns-3-ccod`` directory which presents a deep reinforcement learning scenario with Reinforced-lib and ns-3.