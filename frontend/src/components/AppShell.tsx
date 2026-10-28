import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Activity, ArrowUpRight, BarChart3, ClipboardList, FileHeart, HeartPulse, LayoutDashboard, Menu, ShieldCheck, X } from "lucide-react";
import { getHealth } from "../services/api";

export function AppShell() {
  const [apiReady, setApiReady] = useState<boolean | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let active = true;
    const check = () => getHealth().then(() => active && setApiReady(true)).catch(() => active && setApiReady(false));
    void check();
    const timer = window.setInterval(check, 30000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const closeMenu = () => setMenuOpen(false);

  return (
    <div className="app-frame">
      {menuOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={closeMenu} />}
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><Activity size={19} strokeWidth={2.4} /></div>
          <div><span className="brand-name">MedPredict</span><span className="brand-caption">MODEL WORKSPACE</span></div>
          <button className="icon-button sidebar-close" aria-label="Close navigation" onClick={closeMenu}><X size={19} /></button>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          <NavLink to="/" end onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <LayoutDashboard size={18} /><span>Overview</span>
          </NavLink>
          <NavLink to="/dashboard" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <BarChart3 size={18} /><span>Performance</span>
          </NavLink>
          <NavLink to="/assessment/diabetes" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <Activity size={18} /><span>New assessment</span><ArrowUpRight className="nav-trailing" size={15} />
          </NavLink>
          <NavLink to="/assessment/heart" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <HeartPulse size={18} /><span>Heart disease</span>
          </NavLink>
          <NavLink to="/assessment/liver" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <FileHeart size={18} /><span>Liver disease</span>
          </NavLink>
          <NavLink to="/history" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
            <ClipboardList size={18} /><span>History</span>
          </NavLink>
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <ShieldCheck size={17} />
          <p>Dataset estimates only.<br />Never a medical diagnosis.</p>
        </div>
        <div className="sidebar-footer">
          <span className={`status-dot ${apiReady ? "status-online" : apiReady === false ? "status-offline" : ""}`} />
          <span>{apiReady ? "API connected" : apiReady === false ? "API unavailable" : "Connecting to API"}</span>
          <span className="footer-version">v1.0</span>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button className="icon-button menu-toggle" aria-label="Open navigation" onClick={() => setMenuOpen(true)}><Menu size={20} /></button>
          <div className="breadcrumb"><span>MedPredict</span><span className="breadcrumb-separator">/</span><span className="breadcrumb-current">Diabetes readmission</span></div>
          <div className="topbar-note"><span className="topbar-pulse" /> EDUCATIONAL ENVIRONMENT</div>
        </header>
        <main className="main-content"><Outlet /></main>
        <footer className="page-footer">
          <span>MedPredict · Public dataset demonstration</span>
          <span>Not for clinical decisions</span>
        </footer>
      </div>
    </div>
  );
}