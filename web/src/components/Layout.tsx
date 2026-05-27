import { Outlet, NavLink, Link } from "react-router-dom";
import "../styles/Layout.css";
import { useToastStore } from "../store/layoutStore";
import type { Toast as ToastType } from "../types";

const navItems = [
    { path: "/face_detection", label: "Face Detection" },
    { path: "/face_landmark", label: "Face Landmark" },
    { path: "/face_comparison", label: "Face Comparison" },
    { path: "/face_verification", label: "Face Verification" }
];

function Layout() {
    const {toastList} = useToastStore()
    return (
        <div className="layout-root"> 
            {/* Đưa container ra ngoài làm wrapper bao bọc tất cả toast */}
            {toastList.length > 0 && (
                <div className="toast-container">
                    {toastList.map((v) => <Toast key={v.id} toast={v}/>)}
                </div>
            )}
            
            <Header />
            <main className="layout-content">
                <Outlet />
            </main>
        </div>
    );
}

function Toast({toast} : {toast: ToastType}) {
    const {removeToast} = useToastStore()
    return (
        /* Xóa toast-container ở đây, đổi thành class trạng thái hoặc toast-item */
        <div className={`toast-item ${toast.error ? 'type-error' : 'type-info'}`}>
            <div className="toast-content">
                <div className="toast-header">
                    <div className="toast-title-group">
                        {toast.error ? (
                            <span className="toast-icon icon-error">✕</span>
                        ) : (
                            <span className="toast-icon icon-info">✓</span>
                        )}
                        <h3 className="toast-title">{toast.title == undefined ? "" : toast.title}</h3>
                    </div>
                    <button className="toast-close" onClick={() => removeToast(toast.id)}>×</button>
                </div>
                <div className="toast-body">
                    <p>
                        {toast.message == undefined? "" : toast.message}
                    </p>
                </div>
            </div>
        </div>
    );
}

function Header() {
    return (
        <header className="header"> 
            <Link to="/" className="header-brand">
                <span className="header-brand-dot"></span>
                AI Vision
            </Link>
            
            <nav className="nav">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            isActive ? "nav-link active" : "nav-link"
                        }
                    >
                        {item.label}
                    </NavLink>
                ))}
            </nav>
        </header>
    );
}

export default Layout;