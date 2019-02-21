---
title: Human Switching
subtitle: POMDP Formulation
link-citations: true
numbersections: true
mainfont: Latin Modern Roman
beamercovered: transparent
innertheme: rectangles
titlegraphic: 'media/julia-pomdp.png'
header-includes:
    - \usepackage[export]{adjustbox}
    - \usepackage{chngpage}
    - \usepackage{todonotes}
    - \usepackage{multimedia}
    - \usepackage{multicol}
    - \newcommand{\itodo}[1]{\todo[inline]{#1}}
    - \newcommand{\hideFromPandoc}[1]{#1}
         \hideFromPandoc{
             \let\Begin\begin
             \let\End\end
         }
---

# State Space

**External State (fully observed)**

- human pose $x_H$
- robot pose $x_R$

\pause
\vspace{20pt}

**Internal States (latent)**

- human behavior model with some latent $\theta$
    - $\theta$: *human* reward model, target etc.

# Transition Model and Action Space

**Human Transition**

- $u_H$ sampled from human behavior model
    - **currently:** linear human with hidden target
    - **next:** Boltzmann human

$$
P\left(u^t_H | x^T_H; \beta, \theta \right) \propto e^{\beta Q_H\left(x^t_H, u^t_H; \theta\right)}
$$
$$
x^{t+1}_H = f\left(x^t_H, u_H\right)
$$

\pause

**Robot Transition**

- $u_R$ from discrete action space
    - **currently:** up, down, left, right, stop
    - **maybe later:** other "macro" actions

$$
x^{t+1}_R = f\left(x^t_R, u_R\right)
$$

# POMDP Framework / Setup

1. Belief Updater |  Particle Filter
    - used to estimate latent human behavior model
        - **for now:** infers target of linear human
        - **next:** $\beta, \theta$ of Boltzmann

\pause
\vspace{20pt}

2. POMDP Solver
    - computes conditional plan based on estimated belief
    - using some `MCTS` variant (currently POMCPOW)
