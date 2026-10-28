import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AssessmentPage } from "./pages/AssessmentPage";
import { HeartPage } from "./pages/HeartPage";
import { HistoryPage } from "./pages/HistoryPage";
import { LiverPage } from "./pages/LiverPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PerformanceDashboardPage } from "./pages/PerformanceDashboardPage";
import { ResultPage } from "./pages/ResultPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<OverviewPage />} />
          <Route path="/dashboard" element={<PerformanceDashboardPage />} />
          <Route path="/assessment/diabetes" element={<AssessmentPage />} />
          <Route path="/assessment/heart" element={<HeartPage />} />
          <Route path="/assessment/liver" element={<LiverPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/result/:id" element={<ResultPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}