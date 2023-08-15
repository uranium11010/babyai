import os
import pickle
import numpy as np

from .. import utils
import blosc

from enum import IntEnum

from abstractions.steps import AxStep, Solution
from abstractions.abstractions import Axiom, ABS_TYPES
from abstractions.compress import IAPLogN


def get_demos_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)


def load_demos(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))


def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


def abstract_demos(demos, args, rules=None):
    solutions = []
    for demo in demos:
        all_images = blosc.unpack_array(demo[1])
        directions = demo[2]
        actions = demo[3]
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        states = list(zip(all_images, directions))
        states.append(None)
        actions = [AxStep(action._value_, args.abs_type) for action in actions]
        solutions.append(Solution(states, actions))

    axioms = [Axiom(i, args.abs_type) for i in range(7)]
    if rules is None:
        compressor = IAPLogN(solutions, axioms, args.__dict__)
        _, new_sols = compressor.abstract()
        rules = compressor.new_axioms
    else:
        new_sols = ABS_TYPES[args.abs_type].get_abstracted_sols(solutions, rules[len(axioms):])

    # AbsActions = IntEnum('AbsActions', ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'] +
    #         [f'abs_{i}' for i in range(len(rules)-len(axioms))] + ['break_loop', 'do_nothing'], start=0)
    # import pickle as pkl
    # pkl.dump(AbsActions, open('test.pkl', 'wb'))

    new_demos = []
    for demo, sol in zip(demos, new_sols):
        # assume no nested loops
        new_demos.append(sol_to_demo(sol, demo[0], rules))
    return new_demos, rules


def sol_to_demo(sol, mission, new_axioms):
    # assume no nested loops
    rule2idx = {rule: i for i, rule in enumerate(new_axioms)}
    def rule_to_actions(rule):
        if rule.is_axiom:
            yield 0
        else:
            for sub_rule, num_reps in zip(rule.rules, rule.ex_num_reps):
                if isinstance(sub_rule, tuple):
                    for i in range(num_reps):
                        for sub_sub_rule in sub_rule:
                            yield len(new_axioms) + 1  # continue
                    yield len(new_axioms)
                else:
                    yield len(new_axioms) + 1
    actions = [rule2idx[step.rule] if i == 0 else action
                for step in sol.actions for i, action in enumerate(rule_to_actions(step.rule))]
    # actions = [abs_actions(action) for action in actions]
    all_images = []
    directions = []
    just_broke_loop = False
    for k, step in enumerate(sol.actions):
        if step.rule.is_axiom:
            image, direction = sol.states[k]
            all_images.append(image)
            directions.append(direction)
            if just_broke_loop:
                all_images.append(image)
                directions.append(direction)
                just_broke_loop = False
        else:
            j = 0
            for sub_rule, num_reps in zip(step.rule.rules, step.rule.ex_num_reps):
                if isinstance(sub_rule, tuple):
                    for i in range(num_reps):
                        for sub_sub_rule in sub_rule:
                            image, direction = step.ex_states[j]
                            all_images.append(image)
                            directions.append(direction)
                            j += 1
                            if just_broke_loop:
                                all_images.append(image)
                                directions.append(direction)
                                just_broke_loop = False
                    just_broke_loop = True
                else:
                    image, direction = step.ex_states[j]
                    all_images.append(image)
                    directions.append(direction)
                    j += 1
                    if just_broke_loop:
                        all_images.append(image)
                        directions.append(direction)
                        just_broke_loop = False
    if just_broke_loop:
        actions.pop()
    return mission, blosc.pack_array(np.array(all_images)), directions, actions
