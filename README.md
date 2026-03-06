# 🛡️ Proactive Cognitive Shield: Uncertainty-Aware Offline World Models for Safe HRC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Simulator: CoppeliaSim](https://img.shields.io/badge/Simulator-CoppeliaSim-blue.svg)](https://www.coppeliarobotics.com/)

> **"로봇이 행동하기 전에 인간의 인지적 스트레스를 상상하고, 스스로 마음의 벽을 넓힌다."**
> 본 프로젝트는 뇌파(EEG) 장비 착용 없이, 오프라인 세계 모델(Offline World Model)과 불확실성 추정(Uncertainty Estimation)을 활용하여 인간-로봇 협업(HRC) 환경에서 **사전 예방적 인지 안전(Proactive Cognitive Safety)**을 달성하는 연구입니다.

## 💡 Overview
기존의 뇌파 기반 로봇 안전 제어는 인간이 놀란 후(ErrP 발생)에야 로봇을 정지시키는 **사후 반응형(Reactive)** 시스템에 머물러 있었습니다. 이는 본질적인 심리적 스트레스를 예방하지 못하며, 실시간 뇌파 장비 착용이라는 비실용성을 동반합니다. 

본 연구는 로봇의 저차원 고유수용성 상태(Proprioceptive state)만을 입력받아 인간의 뇌파 반응을 사전에 예측하는 **오프라인 세계 모델**을 제안합니다. 특히, 학습되지 않은 낯선 궤적(OOD)에서 발생하는 위험을 막기 위해 모델의 **예측 불확실성(Epistemic Uncertainty)**을 수치화하고, 이를 기반으로 코펠리아심(CoppeliaSim) 환경 내에서 실시간으로 팽창/수축하는 **동적 인지 방어막(Dynamic Cognitive Shield)**을 구축합니다.

## 🌟 Key Contributions
1. **Sensorless Proactive Safety:** 실제 운영 환경에서 번거로운 뇌파 장비 없이, 사전 학습된 모델의 상상만으로 인지적 스트레스를 예방합니다.
2. **Uncertainty-Aware OOD Defense:** 단일 모델(EDL/MC Dropout)의 불확실성(Variance)을 활용하여, 분포 밖(OOD)의 낯선 궤적을 마주했을 때 로봇이 선제적으로 행동을 기각하도록 설계되었습니다.
3. **Dynamic Virtual Shield:** 물리적 충돌 방지를 넘어, '인간의 스트레스'와 'AI의 불확실성'을 물리적 거리(Distance Constraint)로 치환한 직관적인 제어 방벽 함수(CBF)를 도입했습니다.

## ⚙️ System Architecture & Governing Equation
로봇과 인간 사이의 안전 거리 $d_{safe}$ 는 모델이 예측한 평균 뇌파 스트레스($\mu$)와 예측의 불확실성($\sigma^2$)에 비례하여 동적으로 변합니다.

$$d_{safe}(s, a) = d_{min} + \alpha \cdot \mu_{EEG}(s, a) + \beta \cdot \sigma^2_{EEG}(s, a)$$

- $d_{min}$: 최소 보장 물리적 안전 거리 (Fallback System)
- $\mu_{EEG}$: World Model이 예측한 평균 인지적 스트레스 (ErrP Probability)
- $\sigma^2_{EEG}$: OOD 상황을 감지하는 모델의 예측 분산 (Epistemic Uncertainty)
- $\alpha, \beta$: 각 요소의 반영 가중치 파라미터

## 🧪 Evaluation Strategy (Hold-out Validation)
OOD 상황에 대한 정답(Ground Truth) 부재 문제를 해결하기 위해 **Hold-out 궤적 테스트**를 수행합니다. 
데이터셋 내에서 실제 인간에게 가장 큰 ErrP(스트레스)를 유발했던 위험 궤적을 학습(Train)에서 제외하여 고의적인 OOD 상태로 만듭니다. 이후 테스트(Test) 단계에서 모델이 해당 궤적을 마주했을 때, 불확실성($\sigma^2$) 폭발과 함께 방어막을 팽창시켜 궤적을 사전에 차단하는지 시뮬레이션을 통해 검증합니다.

## 🛠️ Environment & Prerequisites
- **OS:** Ubuntu 22.04 / Windows 11
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch
- **Simulation:** CoppeliaSim (via ZMQ Remote API)
- **Dataset:** Open-source ErrP-HRI Dataset (Ehrlich et al.)

## 📅 Roadmap (TODOs)
- [ ] CoppeliaSim 시뮬레이션 환경 및 인간 아바타 세팅
- [ ] ErrP 데이터셋 다운로드 및 전처리 파이프라인 구축 (Low-dim state 정렬)
- [ ] Uncertainty-aware World Model (MC Dropout / EDL) 아키텍처 설계 및 오프라인 학습
- [ ] CoppeliaSim 내 동적 방어막(Virtual Shield) 렌더링 및 제어 루프 연동
- [ ] Hold-out OOD 궤적 방어 실험 및 결과 그래프(Metric) 도출