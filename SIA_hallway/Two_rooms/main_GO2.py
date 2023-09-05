from ENV.env_two_rooms_single import EnvGoObstacle
from algorithm.Q_learning import Q_learning
import numpy as np


if __name__ == '__main__':
    map_size  = 7
    num_agent = 1
    env = EnvGoObstacle(map_size, num_agent)
    max_episode = 2500
    max_steps = 50
    Q_learning = Q_learning(map_size * map_size, 5)

    for i in range(max_episode):
        state = env.reset()
        count = 0
        done = False
        rewards = 0
        while not done:
            #env.plot_scene()
            # print('former-state:', env.former_states)
            # print('state       :', env.state)
            #env.render()
            # if i > max_episode - 2:
            #     env.render()
            s = state[0][0] * map_size + state[0][1]
            epsilon = np.min([0.95, 0.01+(0.95-0.01)*(i*max_steps+count)/(max_episode*max_steps)])
            action = Q_learning.choose_action(s, epsilon)

            reward, done, next_state = env.step([action])

            if count >= max_steps - 1:
                done = True
            else:
                count += 1
            a = action
            r = reward[0]
            s_ = next_state[0][0]*map_size+next_state[0][1]
            rewards += r
            Q_learning.learn(s,a,r,s_)
            state = next_state
        np.save("li_Q_value2.npy", Q_learning.get_Q_value())
        print(i, count, rewards)