# Plant Disease Detection and Smart Spray Assistant

## 👥 Team Details
- **Team Name:** Codegen
- **Members:** Anoop Narayan (Hardware), Godly K Mathews (Software Backend/UI), Bimal Ghosh (Hardware), Ajin John Thomas (Hardware)
- **Contact Email:** Add team emails
- **GitHub / Portfolio Links:** Add links

## 🚀 Problem Statement
Farmers often detect plant disease late, spray inconsistently, and have no simple history view of what happened in the field.  
The project solves this by combining detection, spray status tracking, and AI guidance so farmers can react quickly and correctly.

## 💡 Solution Overview
This project provides a farmer dashboard that shows detected diseases, spray status, and latest alerts from the field pipeline (ESP32 + Raspberry Pi + backend API).  
From each disease event, farmers can click **Ask Gemma** to get plain-language disease insight and treatment steps using Gemma 4 (local first, cloud fallback).

## 🧩 Key Features
- Live disease event dashboard with refresh every 10 seconds
- Spray status tracking (`Treated` / `Needs care`)
- **Ask Gemma** per disease event
- **Gemma Insight** page for actionable guidance
- Local Gemma endpoint support with cloud Gemini fallback

## ⚙️ Software & AI Stack
- **AI Models / APIs:** Gemma 4 (local gateway), Gemini (cloud fallback)
- **Languages:** TypeScript
- **Frameworks / Libraries:** React 18, Vite, Tailwind CSS
- **Backend API integration:** REST endpoints for health and disease events

## 🧠 AI Integration Details
- **What AI does:** Converts detected disease names into simple farmer guidance (insight + treatment)
- **Why Gemma/Gemini:** Fast practical responses, easy local deployment and cloud backup
- **Runtime mode:** Hybrid (local Gemma first, cloud Gemini fallback)

## 🔌 System Architecture
`Camera/Scanner → Raspberry Pi/ESP32 pipeline → Backend API → React Dashboard → Ask Gemma (AI guidance)`

## 🧪 How It Works (Step-by-Step)
1. Field pipeline posts disease events to backend API.
2. Dashboard fetches and displays latest events with status.
3. Farmer opens **Ask Gemma** on a disease row.
4. UI calls local Gemma gateway; if unavailable, falls back to cloud Gemini.
5. Farmer reads insight and treatment guidance.

## 📊 Results & Performance
- Real-time style updates every 10 seconds
- Clear disease and treatment visibility for faster action
- Reliable AI access through local-first with cloud fallback

## 🌍 Real-World Impact
- Helps farmers make quicker treatment decisions
- Reduces missed spraying and improves crop protection workflow
- Scales to multiple farms with shared API + model infrastructure

## 🔮 Future Improvements
- Add image snapshot preview for each event
- Add multilingual response support
- Add severity scoring and priority alerts

## 📦 Setup Instructions
1. Install dependencies:
   ```bash
   npm install
   ```
2. Create `.env`:
   ```bash
   VITE_API_BASE_URL=http://127.0.0.1:8080
   VITE_GEMMA_BASE_URL=http://192.168.10.249:4000
   VITE_GEMMA_API_KEY=your_gemma_api_key_here
   VITE_GEMMA_MODEL=gemma-local
   VITE_GEMINI_API_KEY=your_google_gemini_key_for_fallback
   ```
3. Run:
   ```bash
   npm run dev
   ```
4. Build:
   ```bash
   npm run build
   ```

## 📎 Additional Resources
- **GitHub Repo:** `godlykmathews/plant-monitoring-UI`
- **Backend API Endpoints Used:**
  - `GET /health`
  - `GET /api/disease-events?limit=20`
  - `POST /api/disease-events`
