# PawCare AI 🐕

**AI-powered street dog disease detection & outbreak tracking system**

[![Live Demo](https://img.shields.io/badge/Live-Vercel-brightgreen)](https://pawcare-frontend-five.vercel.app)
[![Backend API](https://img.shields.io/badge/API-Render-blue)](https://pawcare-backend-eimp.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

PawCare AI is a full-stack machine learning application that detects skin diseases in street dogs from photos. It combines computer vision with real-time outbreak detection to help NGOs, veterinarians, and animal rescue organizations identify and track disease patterns in vulnerable dog populations.

**Problem:** Street dogs suffer from untreated skin diseases. Manual diagnosis requires a vet visit, which is impractical at scale.
**Solution:** Instant AI-powered preliminary diagnosis from a photo, with built-in vet verification, calibrated confidence, and out-of-distribution rejection for safety.

---

## ✨ Features

### 🔍 Disease Detection
- Upload dog photos (JPG/PNG)
- AI predicts disease from 6 classes: Dermatitis, Fungal infections, Healthy, Hypersensitivity, Demodicosis, Ringworm
- Temperature-scaled confidence calibration (T=1.5525) with a data-justified 0.70 minimum confidence threshold
- Out-of-distribution (OOD) detection via Mahalanobis distance on penultimate-layer features — rejects garbage/unrecognizable images even when the classifier itself would report high confidence
- Dog detection gate (ImageNet-based, rejects non-dog photos)

### 👥 Multi-Role Authentication
- **User:** Report cases, view results, request adoptions
- **Vet:** Approve registrations, verify diagnoses, provide corrections
- **Admin:** Manage users, approve vets, export training data, add pets
- JWT stored in httpOnly cookies (not localStorage) with CSRF double-submit protection, persisted in Neon Postgres

### 📋 Case Management
- Complete case history with photo viewing
- Vet confirmation/correction workflow
- Status tracking: pending → vet_confirmed → resolved
- Human-in-the-loop data pipeline (vet corrections become training data)

### 🗺️ Outbreak Detection
- GPS-based clustering (single-linkage, Haversine distance) within a rolling 14-day window
- Evidence-weighted scoring per case (vet-confirmed > confident AI > uncertain AI; "Healthy" cases excluded)
- Cluster fires only with ≥2 cases and weighted score ≥2.0, tagged `confirmed_outbreak` vs `possible_cluster`
- Real-time cluster visualization (Leaflet maps)
- Auto-notifies NGOs, with a per-NGO/per-admin cooldown to prevent notification spam

### 📱 Offline Support
- Case uploads queue in browser (localStorage)
- Auto-syncs when connection returns
- Smart retry logic (distinguishes permanent vs transient failures)

### 🐾 Pet Adoption Portal
- Admin adds dogs to adoption database
- Users browse and request adoption
- NGOs notified and contact adopters directly

### 🌍 NGO Locator & Management
- Map view of nearby verified NGOs, plus live discovery of shelters/vets via OpenStreetMap Overpass
- Automatic outbreak notifications
- Admin-managed verified NGO database

---

## 🏗️ Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Frontend** | React 18 + TypeScript | Vite build, wouter routing, TanStack Query, Tailwind + shadcn/ui, Leaflet maps |
| **Backend** | Flask (Python) | Blueprint-organized routes, JWT auth (cookie-based), CORS, REST API |
| **ML Model** | MobileNetV2 ONNX (dual-output: logits + pooled features) | Transfer learning, 6 disease classes, 96.77% test accuracy |
| **Database** | Neon Postgres | Persistent, free tier |
| **Image Storage** | Cloudinary | Persistent cloud storage, free tier — no local disk fallback |
| **Deployment** | Render + Vercel | Backend on Render, frontend on Vercel (via a rewrite proxy for first-party cookies) |

---

## 🚀 Live Demo

**Try it now:**
- **Frontend:** https://pawcare-frontend-five.vercel.app
- **Backend API:** https://pawcare-backend-eimp.onrender.com

### Demo Credentials
- **Admin:** set via `FIXED_ADMIN_USERNAME` / `FIXED_ADMIN_PASSWORD` env vars
- **Test User:** Create a new account

### Quick Test Flow
1. Sign up as a user
2. Upload a dog photo → see AI prediction (or a graceful "not a dog" / "unable to classify" response)
3. Sign up as a vet → wait for admin approval
4. As vet, confirm/correct diagnoses
5. As admin, export vet corrections
6. Request pet adoption as user

---

## 🏃 Quick Start (Local Development)

**Note:** Frontend and backend are in **separate GitHub repositories**. You need to clone and run both.

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Backend Setup (Separate Repo)

```bash
# Clone backend repo
git clone https://github.com/karthi206/Pawcare-backend.git
cd Pawcare-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Model files already live in model/:
# pawcare_mobilenetv2_with_features.onnx (+ .onnx.data), general_imagenet_model.onnx,
# class_means.npy, cov_inv.npy

# Set environment variables (create a .env file — it's auto-loaded)
JWT_SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///cases.db  # Local SQLite for dev
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
FIXED_ADMIN_USERNAME=admin
FIXED_ADMIN_PASSWORD=set-a-password

# Run Flask
python app.py
```

Backend runs at: `http://localhost:5000`

### Frontend Setup (Separate Repo)

```bash
# Clone frontend repo (in a new terminal/folder)
git clone <frontend-repo-url>
cd pawcare-frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## 📦 Deployment

### Backend Deployment (Render)

1. Push this repo to GitHub
2. In Render dashboard: Connect GitHub repo
3. Set environment variables:
   ```
   DATABASE_URL=postgresql://user:pass@host/dbname  # From Neon
   CLOUDINARY_CLOUD_NAME=your-cloud-name
   CLOUDINARY_API_KEY=your-api-key
   CLOUDINARY_API_SECRET=your-api-secret
   JWT_SECRET_KEY=<generate-strong-random-key>
   FIXED_ADMIN_USERNAME=admin
   FIXED_ADMIN_PASSWORD=<set-password>
   RESEND_API_KEY=<your-resend-key>
   ADMIN_EMAIL=<admin-notification-inbox>
   ```
4. Deploy

**Backend live at:** https://pawcare-backend-eimp.onrender.com

### Frontend Deployment (Vercel)

1. Push `pawcare-frontend` repo to GitHub
2. In Vercel dashboard: Connect GitHub repo
3. Set environment variable:
   ```
   VITE_API_URL=/api
   ```
   (Requests are routed through Vercel's rewrite proxy so the httpOnly auth cookie stays first-party — do not point this at the Render URL directly.)
4. Deploy

**Frontend live at:** https://pawcare-frontend-five.vercel.app

---

## 📊 Model Performance

- **Architecture:** MobileNetV2 transfer learning, dual-output ONNX export (class logits + 1280-dim pooled features)
- **Dataset:** Kaggle "Dog's Skin Diseases Image Dataset" — 4,315 images across 6 classes, predefined train/valid/test split (3022/860/433)
- **Test accuracy:** 96.77% (419/433) on the held-out test set, per-class F1 ranging 0.93–0.995
- **Calibration:** Temperature scaling (T=1.5525), fit via LBFGS on validation logits
- **Confidence threshold:** 0.70, chosen because test-set accuracy was only ~75% below this threshold vs. 98%+ at/above it
- **OOD detection:** Mahalanobis distance on pooled features vs. per-class training means (threshold 75.0) — catches inputs the classifier would otherwise confidently mislabel
- **Inference time:** <100ms per image (ONNX Runtime)

### Limitations
- Even at 96.77% accuracy, the model still misclassifies confidently in a minority of cases — vet verification remains the safety net, not a formality
- Model trained only on the six specific dog skin conditions in the dataset — no independent external validation dataset has been benchmarked yet
- Not suitable for real-time triage without veterinary oversight

---

## 🔐 Security Considerations

- ✅ JWT authentication in httpOnly cookies (not localStorage) with CSRF double-submit protection, 24h expiry
- ✅ Role-based access control (user/vet/admin), enforced server-side on every protected route
- ✅ Secure password hashing
- ✅ CORS configured for trusted origins only
- ✅ SQL injection protection via SQLAlchemy ORM
- ✅ Image upload validation: extension allowlist, MIME check, Pillow verification, dimension bounds
- ✅ Cloudinary-only image storage — no local disk fallback (returns 502 if upload fails, rather than persisting locally)
- ✅ Per-endpoint cooldown on NGO outbreak notifications
- ⚠️ No general-purpose rate limiting across the API (e.g. login/register are not throttled) — recommended before scaling beyond a demo/portfolio deployment

---

## 🛠️ API Endpoints

### Authentication
- `POST /auth/register` — Create new user/vet account
- `POST /auth/login` — Login
- `POST /auth/logout` — Logout
- `GET /auth/me` — Get current user
- `PATCH /auth/profile` — Update profile
- `PATCH /auth/password` — Change password

### Cases
- `POST /upload` — Upload dog photo & get diagnosis
- `GET /uploads/<filename>` — Serve an uploaded image
- `GET /cases` — List cases (auth required)
- `GET /cases/<id>` — Get case details (auth required)
- `PATCH /cases/<id>/status` — Vet confirms/corrects diagnosis
- `GET /clusters` — Get outbreak clusters

### Pets (Adoption)
- `GET /pets` — List available pets
- `POST /pets` — Admin adds pet (requires auth)
- `POST /pets/<id>/adopt` — User requests adoption

### Admin
- `GET /admin/pending-vets` — List pending vet applications
- `POST /admin/vets/<id>/approve` — Approve vet (sends decision email)
- `POST /admin/vets/<id>/reject` — Reject vet (sends decision email)
- `GET /admin/export-corrections` — Export vet-corrected cases (for retraining)

### NGOs
- `GET /ngos` — List verified NGOs
- `GET /ngos/nearby` — Nearby verified NGOs (Haversine distance)
- `GET /ngos/live-nearby` — Live shelter/vet/NGO discovery via OpenStreetMap Overpass
- `POST /ngos` — Admin creates NGO
- `POST /ngos/<id>/notify` — Notify NGO of outbreak (cooldown-limited)

### Geocoding
- `GET /geocode/reverse` — Reverse geocode coordinates to an address

Every route above is also available under an `/api/...` prefix (e.g. `/api/auth/login`), for environments where the frontend proxy expects an `/api` base path.

---

## 📁 Project Structure

### Backend Repo (`Pawcare-backend`)
```
Pawcare-backend/
├── app.py                     # Flask app factory: config, CORS, JWT error handlers, blueprint registration
├── config.py                  # Env-backed Config class, .env loading, JWT secret enforcement
├── extensions.py              # Shared db / jwt extension instances
├── helpers.py                 # Shared helper functions
├── ml.py                      # Loads ONNX models + OOD reference arrays at startup
├── models.py                  # SQLAlchemy models (Case, User, NGO, NGONotification, Pet, AdoptionRequest)
├── clustering.py              # Outbreak detection (weighted, time-windowed Haversine clustering)
├── email_service.py           # Resend API integration (vet registration/decision emails)
├── clean_duplicate_cases.py   # Standalone dedup script (dry-run by default, --execute to apply)
├── startup.py                 # DB migrations + fixed-admin seeding, run at app startup
├── requirements.txt
├── routes/
│   ├── auth.py                 # Register/login/logout/me/profile/password
│   ├── cases.py                # Upload, case listing, status updates, clusters
│   ├── ngos.py                 # NGO CRUD, nearby search, live discovery, notify
│   ├── pets.py                 # Pet listing, adoption requests
│   ├── admin.py                # Vet approval workflow, corrections export
│   └── geocode.py              # Reverse geocoding
├── services/
│   ├── geocoding.py             # Geocoding provider integration
│   └── osm.py                   # OpenStreetMap Overpass integration
├── utils/
│   ├── geo.py                   # Haversine + geo helpers
│   └── validation.py            # Image/input validation
├── model/
│   ├── pawcare_mobilenetv2_with_features.onnx (+ .onnx.data)  # Disease detector, dual-output
│   ├── general_imagenet_model.onnx                            # Dog detection gate
│   ├── class_means.npy / cov_inv.npy                          # Mahalanobis OOD reference
│   └── cnn_model.py                                            # Model loading & inference logic
├── scripts/archive/            # Legacy one-off scripts, kept out of active code
├── uploads/                    # Temp folder for image processing
└── README.md
```

### Frontend Repo (`pawcare-frontend`)
```
pawcare-frontend/
├── client/src/
│   ├── components/
│   │   ├── CaseMap.tsx             # Outbreak clustering map (Leaflet)
│   │   ├── ErrorState.tsx          # Reusable error UI
│   │   ├── RequireAuth.tsx         # Route guard for logged-in-only pages
│   │   ├── layout/AppLayout.tsx    # Nav, header/footer
│   │   └── ui/                     # shadcn/ui primitives
│   ├── pages/
│   │   ├── Landing.tsx, DiseaseDetection.tsx, NgoLocator.tsx,
│   │   ├── CaseTracker.tsx, AdoptionPortal.tsx, AdminPanel.tsx,
│   │   └── Login.tsx, Register.tsx, not-found.tsx
│   ├── hooks/
│   │   ├── use-auth.tsx            # Auth context & cookie-based session state
│   │   └── use-toast.ts
│   ├── lib/
│   │   ├── api-client.ts           # Fetch wrapper: cookies, CSRF header, retries, single-flight dedup
│   │   ├── config.ts                # API_URL (relative /api by default)
│   │   └── queryClient.ts, utils.ts
│   ├── App.tsx                      # Router, RequireAuth-wrapped routes
│   └── main.tsx
├── components.json, tsconfig.json, tailwind.config.ts, postcss.config.js
├── vite.config.ts, vercel.json (rewrite proxy config)
└── package.json
```

---

## 🚧 Known Limitations

| Issue | Status | Workaround |
|---|---|---|
| No password reset | Not implemented | Contact admin |
| No email verification | Not implemented | Manual approval workflow |
| General API rate limiting missing | Security gap (NGO notify has a cooldown; other routes don't) | Monitor API usage |
| No independent external validation dataset benchmarked | ML gap | Sourcing plan drafted, not yet executed |
| Mobile app not available | Out of scope | Responsive web design works well |
| Model retraining manual | Not automated | Export vet corrections, retrain in Colab |

---

## 🚀 Future Enhancements

- Mobile app (React Native) with camera integration
- Automated model retraining pipeline
- Real-time push notifications
- Vet clinic dashboard for bulk case management
- Donation integration to fund NGO work
- Multi-language support
- PDF report generation
- General-purpose API rate limiting

---

## 💡 How the ML Pipeline Works

1. **User uploads dog photo**
2. **Dog detection gate** (ImageNet ONNX) → rejects non-dogs with a 422 error
3. **Disease detector** (dual-output MobileNetV2 ONNX) → predicts disease + raw confidence + pooled features
4. **Out-of-distribution check** (Mahalanobis distance on pooled features) → flags inputs too far from the training distribution as `not_recognized`, before classification is trusted
5. **Temperature-scaled calibration + 0.70 confidence threshold** → flags uncertain predictions as `unable_to_classify` rather than guessing
6. **Case saved** to Neon database with Cloudinary photo URL
7. **Vet verification** → vet confirms/corrects the AI prediction
8. **Human-in-the-loop** → vet corrections exported as training data
9. **Model improves** → retrain on vet-corrected cases in Colab

---

## 📝 License

MIT License — see LICENSE file

---

## 👤 Author

**Karthi** — 3rd year CS student

---

## 🙏 Acknowledgments

- **MobileNetV2** transfer learning architecture
- **ONNX Runtime** for optimized inference
- **Neon** for free Postgres hosting
- **Cloudinary** for free image CDN
- **Render** for free backend hosting
- **Vercel** for free frontend hosting
- **OpenStreetMap / Overpass API** for live NGO discovery
- Street dog rescue organizations for inspiration

---

## 📧 Questions or Issues?

Open an issue on GitHub or reach out directly.
