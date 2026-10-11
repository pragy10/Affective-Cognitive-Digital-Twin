import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import apiRouter from "./routes/api.js";

dotenv.config({ path: "../../.env" });

const app = express();
const PORT = process.env.PORT || 5000;

// ── Middleware ───────────────────────────────────────────────────────────────
app.use(cors({ origin: "http://localhost:5173" })); // Vite dev server
app.use(express.json());

// ── MongoDB connection (optional — gracefully degrades if not running) ───────
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/acdt";
mongoose
  .connect(MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.warn("MongoDB not connected (running without DB):", err.message));

// ── Routes ───────────────────────────────────────────────────────────────────
app.use("/api", apiRouter);

// ── Health check ─────────────────────────────────────────────────────────────
app.get("/", (req, res) => res.json({ status: "ACDT API running" }));

app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));
