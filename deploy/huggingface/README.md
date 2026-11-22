---
title: SkinScan API
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Backend for SkinScan.AI: a FastAPI service running a DenseNet121 + patient-metadata model, Grad-CAM,
and ABCD scoring. It is decision support for clinicians, not a diagnosis.

Health check: `/api/health`. The web app is hosted separately and calls this API.
