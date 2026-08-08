# PawCare AI 🐕

**AI-powered street dog disease detection & outbreak tracking system**

[![Live Demo](https://img.shields.io/badge/Live-Vercel-brightgreen)](https://pawcare-frontend-azure.vercel.app)
[![Backend API](https://img.shields.io/badge/API-Render-blue)](https://pawcare-backend-eimp.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

PawCare AI is a full-stack machine learning application that detects skin diseases in street dogs from photos. It combines computer vision with real-time outbreak detection to help NGOs, veterinarians, and animal rescue organizations identify and track disease patterns in vulnerable dog populations.

**Problem:** Street dogs suffer from untreated skin diseases. Manual diagnosis requires a vet visit, which is impractical at scale.  
**Solution:** Instant AI-powered preliminary diagnosis from a photo, with built-in vet verification for accuracy.

---

## ✨ Features

### 🔍 Disease Detection
- Upload dog photos (JPG/PNG)
- AI predicts disease from 6 classes: Dermatitis, Fungal infections, Healthy, Hypersensitivity, Demodicosis, Ringworm
- Confidence scoring with safety threshold (60% min)
- Dog detection gate (rejects non-dog photos)

### 👥 Multi-Role Authentication
- **User:** Report cases, view results, request adoptions
- **Vet:** Approve registrations, verify diagnoses, provide corrections
- **Admin:** Manage users, approve vets, export training data, add pets
- JWT-based, persisted in Neon Postgres

### 📋 Case Management
- Complete case history with photo viewing
- Vet confirmation/correction workflow
- Status tracking: pending → vet_confirmed → resolved
- Human-in-the-loop data pipeline (vet corrections become training data)

### 🗺️ Outbreak Detection
- GPS-based clustering (Haversine distance)
- Groups cases within 1km radius, same disease, min 2 cases
- Real-time cluster visualization (Leaflet maps)
- Auto-notifies NGOs

### 📱 Offline Support
- Case uploads queue in browser (localStorage)
- Auto-syncs when connection returns
- Smart retry logic (distinguishes permanent vs transient failures)

### 🐾 Pet Adoption Portal
- Admin adds dogs to adoption database
- Users browse and request adoption
- NGOs notified and contact adopters directly

### 🌍 NGO Locator & Management
- Map view of nearby NGOs
- Automatic outbreak notifications
- Admin-managed NGO database

---

## 🏗️ Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Frontend** | React 18 + TypeScript | Vite build, Shadcn UI, Leaflet maps |
| **Backend** | Flask (Python) | JWT auth, CORS, REST API |
| **ML Model** | MobileNetV2 ONNX | Transfer learning, 6 disease classes, ~88% accuracy |
| **Database** | Neon Postgres | Persistent, free tier |
| **Image Storage** | Cloudinary | Persistent cloud storage, free tier |

| **Deployment** | Render + Vercel | Backend on Render, frontend on Vercel |

---

## 🚀 Live Demo

**Try it now:**
- **Frontend:** https://pawcare-frontend-azure.vercel.app
- **Backend API:** https://pawcare-backend-eimp.onrender.com

### Demo Credentials
- **Admin:** `admin` / `admin123` (or set your own)
- **Test User:** Create a new account

### Quick Test Flow
1. Sign up as a user
2. Upload a dog photo → see AI prediction
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
git clone <backend-repo-url>
cd pawcare-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (in model/ folder)
# Models should be: pawcare_model.onnx, general_imagenet_model.onnx
# (Download from your training Colab or cloud storage)

# Set environment variables
export FLASK_ENV=development
export JWT_SECRET_KEY=your-secret-key-here
export DATABASE_URL=sqlite:///cases.db  # Local SQLite for dev

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

1. Push `pawcare-backend` repo to GitHub
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
   ```
4. Deploy

**Backend live at:** https://pawcare-backend-eimp.onrender.com

### Frontend Deployment (Vercel)

1. Push `pawcare-frontend` repo to GitHub
2. In Vercel dashboard: Connect GitHub repo
3. Set environment variable:
   ```
   VITE_API_URL=https://pawcare-backend-eimp.onrender.com
   ```
4. Deploy

**Frontend live at:** https://pawcare-frontend-azure.vercel.app

---



## 📊 Model Performance

- **Architecture:** MobileNetV2 transfer learning
- **Dataset:** 200+ images across 6 disease classes
- **Accuracy:** ~88% on test set
- **Inference Time:** <100ms per image (ONNX)
- **Memory:** 85MB (optimized from 700MB PyTorch)

### Limitations
- 88% accuracy means occasional misclassifications on blurry/edge-case photos
- Model trained only on specific dog skin conditions
- Always recommend vet verification for medical decisions
- Not suitable for real-time triage without veterinary oversight

---

## 🔐 Security Considerations

- ✅ JWT authentication with expiration
- ✅ Role-based access control (user/vet/admin)
- ✅ Secure password hashing (bcrypt)
- ✅ CORS configured for trusted origins only
- ⚠️ SQL injection protection via SQLAlchemy ORM
- ⚠️ Rate limiting not yet implemented (recommended for production)

---

## 🛠️ API Endpoints

### Authentication
- `POST /auth/register` — Create new user/vet account
- `POST /auth/login` — Login
- `GET /auth/me` — Get current user

### Cases
- `POST /upload` — Upload dog photo & get diagnosis
- `GET /cases` — List all cases
- `GET /cases/<id>` — Get case details
- `PATCH /cases/<id>/status` — Vet confirms/corrects diagnosis
- `GET /clusters` — Get outbreak clusters

### Pets (Adoption)
- `GET /pets` — List available pets
- `POST /pets` — Admin adds pet (requires auth)
- `POST /pets/<id>/adopt` — User requests adoption

### Admin
- `GET /admin/pending-vets` — List pending vet applications
- `POST /admin/vets/<id>/approve` — Approve vet
- `POST /admin/vets/<id>/reject` — Reject vet
- `GET /admin/export-corrections` — Export vet-corrected cases (for retraining)

### NGOs
- `GET /ngos` — List NGOs
- `POST /ngos` — Admin creates NGO
- `POST /ngos/<id>/notify` — Notify NGO of outbreak

---

## 📁 Project Structure

### Backend Repo (`pawcare-backend`)
```
pawcare-backend/
├── app.py                     # Flask app & all routes
├── models.py                  # SQLAlchemy models
├── clustering.py              # Outbreak detection (Haversine)
├── requirements.txt           # Python dependencies
├── model/
│   ├── pawcare_model.onnx           # Trained disease detector (MobileNetV2)
│   ├── general_imagenet_model.onnx  # Dog detection gate
│   └── cnn_model.py           # Model loading & inference logic
├── uploads/                   # Temp folder for image processing
└── README.md
```

### Frontend Repo (`pawcare-frontend`)
```
pawcare-frontend/
├── src/
│   ├── components/
│   │   ├── CaseTracker.tsx        # Case history & vet review UI
│   │   ├── AdoptionPortal.tsx     # Pet adoption listing
│   │   ├── AdminPanel.tsx         # Admin controls (add pets, approve vets)
│   │   ├── CaseMap.tsx            # Outbreak clustering map (Leaflet)
│   │   └── ... (other UI components)
│   ├── hooks/
│   │   └── use-auth.ts            # Auth context & JWT token management
│   ├── lib/
│   │   └── config.ts              # API URLs & constants
│   ├── App.tsx                    # Main app router
│   └── main.tsx                   # Entry point
├── index.html
├── package.json
├── vite.config.ts
└── README.md
```

---

## 🚧 Known Limitations

| Issue | Status | Workaround |
|---|---|---|
| No password reset | Not implemented | Contact admin |
| No email verification | Not implemented | Manual approval workflow |
| Rate limiting missing | Security gap | Monitor API usage |
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

---

## 💡 How the ML Pipeline Works

1. **User uploads dog photo**
2. **Dog detection gate** (ImageNet ONNX) → rejects non-dogs with 422 error
3. **Disease detector** (MobileNetV2 ONNX) → predicts disease + confidence
4. **Safety threshold** (60% min) → flags uncertain predictions
5. **Case saved** to Neon database with Cloudinary photo URL
6. **Vet verification** → vet confirms/corrects the AI prediction
7. **Human-in-the-loop** → vet corrections exported as training data
8. **Model improves** → retrain on vet-corrected cases in Colab

---

## 📝 License

MIT License — see LICENSE file

---

## 👤 Author

**Karthi** — 3rd year CS student  
Built: August 2026

---

## 🙏 Acknowledgments

- **MobileNetV2** transfer learning architecture
- **ONNX Runtime** for optimized inference
- **Neon** for free Postgres hosting
- **Cloudinary** for free image CDN
- **Render** for free backend hosting
- **Vercel** for free frontend hosting
- Street dog rescue organizations for inspiration

---

## 📧 Questions or Issues?

Open an issue on GitHub or reach out directly.

**The app is live and fully functional — try it now!**
