import reinforced_lib as rfl
from reinforced_lib.agents.wifi import ParticleFilter
from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA

if __name__ == '__main__':
    rl = rfl.RLib(
        agent_type=ParticleFilter,
        ext_type=IEEE_802_11_ax_RA
    )

    print(rl.observation_space)

    observations = {
        'time': 0.0,
        'n_successful': 0,
        'n_failed': 0,
        'power': 16.0,
        'cw': 15
    }

    action = rl.sample(**observations)
    print(action)

    observations = {
        'time': 0.0,
        'n_successful': 10,
        'n_failed': 0,
        'power': 16.0,
        'cw': 15
    }

    action = rl.sample(**observations)
    print(action)
