"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Reference:
    Foret et al. "Sharpness-Aware Minimization for Efficiently Improving
    Generalization." ICLR 2021. arXiv:2010.01412

핵심 아이디어:
    일반 SGD/AdamW는 loss의 **한 점** 최솟값을 찾음 → sharp minima에 빠질 수 있음.
    SAM은 **주변 rho 반경 내에서 loss 최댓값이 가장 작은 지점**을 찾음 → flat minima.

사용 패턴 (학습 루프):
    for batch in loader:
        # --- 1st step: ascent to "worst-case" nearby point ---
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # --- 2nd step: actual gradient at ascent point ---
        loss = criterion(model(x), y)   # 반드시 재계산
        loss.backward()
        optimizer.second_step(zero_grad=True)

주의:
    - forward-backward를 2회 실행 → 학습 시간 ~1.7×.
    - GradScaler(fp16 AMP)와 결합 시 numeric 이슈 가능. bfloat16 autocast 권장.
    - BatchNorm 통계 이중 업데이트 방지를 위해 내부적으로 forward는 eval()로 감싸지
      않음 (원 논문 구현도 동일). 배치 크기가 작을 때만 영향 있음.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Foret 2021 SAM. base_optimizer(AdamW 등)를 감싸는 wrapper.

    rho: neighborhood 반경 (논문 default 0.05, small-data fine-tune은 0.05-0.1 권장).
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        # base optimizer는 같은 param_groups를 공유
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """현재 gradient 방향으로 rho만큼 '올라가서(ascent)' worst-case 지점으로 이동."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)                   # 파라미터 perturb
                self.state[p]["e_w"] = e_w    # 나중에 되돌리기 위해 저장
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """perturbed 지점에서 계산된 gradient로 원래 위치에서 실제 step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # 원래 위치 복귀
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """전통적 optimizer.step() 호출 인터페이스. closure 필수 (loss 재계산용)."""
        assert closure is not None, "SAM.step() requires closure returning loss."
        closure = torch.enable_grad()(closure)  # closure 안에서는 grad 필요
        loss = closure()          # 1st forward-backward
        self.first_step(zero_grad=True)
        closure()                 # 2nd forward-backward
        self.second_step()
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """모든 param group의 gradient l2 norm (SAM ascent 크기 계산용)."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norms.append(p.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
