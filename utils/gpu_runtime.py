from __future__ import annotations

import gc


def cleanup_gpu_resources() -> None:
    """统一清理 GPU 显存与统计，供训练脚本和 Web 运行时复用。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, "synchronize"):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
    except ImportError:
        return
    except Exception:
        return
