def warmup(app, det_size: int):
    """
    One-time pre-runs to avoid first-call stalls from:
      - ONNXRuntime/InsightFace (detector engines, CUDA context)
      - NumPy BLAS/LAPACK (linalg init for lstsq/SVD)
      - OpenCV CUDA (optional)
    """
    import numpy as np, cv2

    # A) InsightFace / ONNXRuntime warm-up (2 passes at your det_size)
    h = w = int(det_size)
    dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    for _ in range(2):
        try:
            _ = app.get(dummy)
        except Exception:
            pass

    # B) BLAS/LAPACK warm-up: matmul, SVD, and LSTSQ (matches your affine solve)
    A = np.random.randn(128, 128)
    _ = A @ A
    _ = np.linalg.svd(A, full_matrices=False)
    Xh = np.random.randn(16, 4)
    Y  = np.random.randn(16, 3)
    _ = np.linalg.lstsq(Xh, Y, rcond=None)

    # C) OpenCV CUDA warm-up (harmless if CUDA not present)
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            g = cv2.cuda_GpuMat()
            g.upload(dummy)
            _ = cv2.cuda.resize(g, (w//2, h//2))
    except Exception:
        pass
