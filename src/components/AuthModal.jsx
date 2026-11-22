import React, { useState } from 'react';
import { X, Lock, Mail, User, Stethoscope, ShieldCheck, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import { authLogin, authRegister } from '../utils/api';

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'register'
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    role: 'clinician',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  if (!isOpen) return null;

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    setErrorMsg(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      if (activeTab === 'login') {
        const res = await authLogin({
          email: formData.email,
          password: formData.password,
        });
        setSuccessMsg(`Welcome back, ${res.user.name}`);
        setTimeout(() => {
          onAuthSuccess(res.user);
          onClose();
        }, 500);
      } else {
        const res = await authRegister({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          role: formData.role,
        });
        setSuccessMsg(`Account created for ${res.user.name}`);
        setTimeout(() => {
          onAuthSuccess(res.user);
          onClose();
        }, 500);
      }
    } catch (err) {
      setErrorMsg(err.message || 'Authentication failed. Please verify credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoFill = () => {
    setFormData({
      name: 'Dr. Sarah Lin, MD',
      email: 'dr.lin@skinscan.ai',
      password: 'clinical_demo_pass',
      role: 'clinician',
    });
    setErrorMsg(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 pt-[10vh] bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-md rounded-2xl bg-derma-card border border-derma-border shadow-2xl overflow-hidden p-6 sm:p-8 space-y-6">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-xl text-slate-400 hover:text-slate-50 hover:bg-derma-surface transition-all"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-derma-accent p-0.5 shadow-glow-cyan">
            <div className="w-full h-full bg-derma-card rounded-[14px] flex items-center justify-center">
              <Stethoscope className="w-6 h-6 text-derma-accent" />
            </div>
          </div>
          <h2 className="text-xl font-bold text-slate-50 tracking-tight">
            SkinScan Clinical Workspace
          </h2>
          <p className="text-xs text-derma-muted">
            {activeTab === 'login'
              ? 'Sign in to access persistent scan history & MLOps feedback'
              : 'Register as an authorized clinician or health researcher'}
          </p>
        </div>

        {/* Tab Toggle */}
        <div className="flex rounded-xl bg-derma-surface/80 p-1 border border-derma-border">
          <button
            type="button"
            onClick={() => {
              setActiveTab('login');
              setErrorMsg(null);
            }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'login'
                ? 'bg-derma-accent text-slate-950 shadow-sm font-bold'
                : 'text-slate-400 hover:text-slate-50'
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('register');
              setErrorMsg(null);
            }}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'register'
                ? 'bg-derma-accent text-slate-950 shadow-sm font-bold'
                : 'text-slate-400 hover:text-slate-50'
            }`}
          >
            Register Clinician
          </button>
        </div>

        {/* Error / Success Notifications */}
        {errorMsg && (
          <div className="flex items-center space-x-2 p-3 text-xs text-rose-300 bg-rose-950/40 border border-rose-500/30 rounded-xl animate-shake">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
        {successMsg && (
          <div className="flex items-center space-x-2 p-3 text-xs text-emerald-300 bg-emerald-950/40 border border-emerald-500/30 rounded-xl">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {activeTab === 'register' && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Full Name</label>
              <div className="relative">
                <User className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  placeholder="Dr. Sarah Lin, MD"
                  className="w-full pl-10 pr-4 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-slate-50 placeholder-slate-500 focus:outline-none focus:border-derma-accent transition-colors"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="clinician@hospital.org"
                className="w-full pl-10 pr-4 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-slate-50 placeholder-slate-500 focus:outline-none focus:border-derma-accent transition-colors"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength={6}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-slate-50 placeholder-slate-500 focus:outline-none focus:border-derma-accent transition-colors"
              />
            </div>
          </div>

          {activeTab === 'register' && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Clinical Role</label>
              <select
                name="role"
                value={formData.role}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-derma-surface border border-derma-border rounded-xl text-xs text-slate-50 focus:outline-none focus:border-derma-accent transition-colors"
              >
                <option value="clinician">Clinician / Dermatologist</option>
                <option value="patient">Patient / Consumer</option>
                <option value="researcher">Clinical AI Researcher</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 text-xs font-bold text-slate-950 bg-derma-accent hover:brightness-110 rounded-xl shadow-glow-cyan transition-all transform active:scale-95 disabled:opacity-50"
          >
            {isLoading
              ? 'Authenticating...'
              : activeTab === 'login'
                ? 'Sign In to Clinical Workspace'
                : 'Create Account & Sign In'}
          </button>
        </form>

        {/* Demo Prefill & Guest Mode Actions */}
        <div className="pt-2 border-t border-derma-border/60 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs">
          <button
            type="button"
            onClick={handleDemoFill}
            className="text-derma-accent hover:underline flex items-center space-x-1"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Use Demo Clinician Credentials</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200"
          >
            Continue as Guest (Zero-Storage)
          </button>
        </div>
      </div>
    </div>
  );
}

