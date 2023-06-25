#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import importlib
import random

import numpy as np
import torch
# TODO: this ugly
import sys
sys.path.append('../')
from TUBSRobustCheck.robustness.attacks import ATTACKS

class Attack:
    SUPPORTED_ATTACKS = ["fgsm", "llcm", "metzen", "metzen_uap", "pgd"]
    def __init__(self, epsilon, model, type="fgsm", target_type='dynamic', iterations=60):
        self.epsilon = epsilon
        self.model = model
        self.type = type
        self.target_type = target_type
        self.iterations = iterations
        self._subset_valid()
        self.attack = self._set_attacks()

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def compute_perturbation(self, dataloader):
        self.attack.compute_perturbation(self.model, dataloader)

    def _subset_valid(self):
        assert self.type in self.SUPPORTED_ATTACKS, \
        f"Attack type {self.type} is not supported. Please choose the supported attacks."
    
    def _set_attacks(self):
        if self.type in ["fgsm", "llcm"]:
            attack = self._create_adversarial_attack("fgsm")(epsilon=self.epsilon)
        elif self.type == "metzen":
            attack = self._create_adversarial_attack(self.type)(epsilon=self.epsilon, iterations=self.iterations, target_type=self.target_type, alpha=2 / 255)
        elif self.type == "metzen_uap":
            attack = self._create_adversarial_attack(self.type)(epsilon=self.epsilon, iterations=40, target_type=self.target_type, alpha=2 / 255)
        return attack

    def _create_adversarial_attack(self, name):
        # print(ATTACKS) #debugger
        module = importlib.import_module(ATTACKS[name].classpath)
        attack = getattr(module, ATTACKS[name].classname)
        return attack

    def generate(self, x, y=None):
        with torch.enable_grad():
            if self.type in ["fgsm", "llcm"]:
                targetted = True if self.type == "llcm" else False
                x_adv = self.attack(x, self.model, y, targetted=targetted)
            elif self.type == "metzen":
                x_adv = self.attack(x, self.model)
            elif self.type == "metzen_uap":
                x_adv = self.attack(x, self.model)

        if type(x_adv) == tuple:
            x_adv, target = x_adv
        else:
            target = None

        return x_adv.detach(), target

def main():
    pass


if __name__ == "__main__":
    main()
