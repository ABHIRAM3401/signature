import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import './LoginPage.css';

function LoginPage() {
    const navigate = useNavigate();
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: ''
    });
    const [signatureFile, setSignatureFile] = useState(null);
    const [signaturePreview, setSignaturePreview] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
        setError('');
    };

    const handleSignatureChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            if (!file.type.startsWith('image/')) {
                setError('Please upload an image file');
                return;
            }
            if (file.size > 5 * 1024 * 1024) { // 5MB limit
                setError('Image must be less than 5MB');
                return;
            }
            setSignatureFile(file);
            setError('');

            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setSignaturePreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            if (isLogin) {
                // Login flow
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: formData.email,
                        password: formData.password
                    })
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.message || 'Login failed');

                localStorage.setItem('signatureguard-token', data.token);
                localStorage.setItem('signatureguard-user', JSON.stringify(data.user));
                navigate('/dashboard');
            } else {
                // Registration flow with signature
                if (!signatureFile) {
                    throw new Error('Please upload your signature');
                }

                // Convert signature to base64
                const reader = new FileReader();
                reader.readAsDataURL(signatureFile);

                reader.onloadend = async () => {
                    try {
                        const response = await fetch('/api/auth/register', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                ...formData,
                                signature: reader.result
                            })
                        });

                        const data = await response.json();
                        if (!response.ok) throw new Error(data.message || 'Registration failed');

                        localStorage.setItem('signatureguard-token', data.token);
                        localStorage.setItem('signatureguard-user', JSON.stringify(data.user));
                        navigate('/dashboard');
                    } catch (err) {
                        setError(err.message);
                        setLoading(false);
                    }
                };
                return; // Exit early, onloadend will handle the rest
            }
        } catch (err) {
            setError(err.message);
        } finally {
            if (isLogin) setLoading(false);
        }
    };

    const toggleMode = () => {
        setIsLogin(!isLogin);
        setError('');
        setFormData({ name: '', email: '', password: '' });
        setSignatureFile(null);
        setSignaturePreview(null);
    };

    const removeSignature = () => {
        setSignatureFile(null);
        setSignaturePreview(null);
    };

    return (
        <div className="login-page">
            <ThemeToggle />

            <div className="login-container">
                <div className="login-card">
                    {/* Logo and Header */}
                    <div className="login-header">
                        <div className="logo">
                            <span className="logo-icon">🛡️</span>
                            <h1>SignatureGuard</h1>
                        </div>
                        <p className="tagline">Secure Signature Verification System</p>
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="login-form">
                        <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>

                        {error && (
                            <div className="error-message">
                                <span>⚠️</span> {error}
                            </div>
                        )}

                        {!isLogin && (
                            <div className="input-group">
                                <label htmlFor="name">Full Name</label>
                                <input
                                    type="text"
                                    id="name"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleChange}
                                    placeholder="John Doe"
                                    required={!isLogin}
                                />
                            </div>
                        )}

                        <div className="input-group">
                            <label htmlFor="email">Email Address</label>
                            <input
                                type="email"
                                id="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                placeholder="you@example.com"
                                autoComplete="email"
                                required
                            />
                        </div>

                        <div className="input-group">
                            <label htmlFor="password">Password</label>
                            <div className="password-wrapper">
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    id="password"
                                    name="password"
                                    value={formData.password}
                                    onChange={handleChange}
                                    placeholder="••••••••"
                                    autoComplete={isLogin ? 'current-password' : 'new-password'}
                                    required
                                    minLength={6}
                                />
                                <button
                                    type="button"
                                    className="password-toggle"
                                    onClick={() => setShowPassword(!showPassword)}
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    {showPassword ? '👁️' : '👁️‍🗨️'}
                                </button>
                            </div>
                        </div>

                        {/* Signature Upload - Only for Registration */}
                        {!isLogin && (
                            <div className="input-group">
                                <label>Your Signature</label>
                                <p className="input-hint">Upload a clear image of your signature for verification</p>

                                {signaturePreview ? (
                                    <div className="signature-preview">
                                        <img src={signaturePreview} alt="Signature preview" />
                                        <button
                                            type="button"
                                            className="remove-signature"
                                            onClick={removeSignature}
                                            aria-label="Remove signature"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                ) : (
                                    <div className="signature-upload">
                                        <input
                                            type="file"
                                            id="signature"
                                            accept="image/*"
                                            onChange={handleSignatureChange}
                                            className="file-input"
                                            required={!isLogin}
                                        />
                                        <label htmlFor="signature" className="file-label">
                                            <span className="upload-icon">📝</span>
                                            <span>Click to upload signature</span>
                                            <span className="file-hint">PNG, JPG up to 5MB</span>
                                        </label>
                                    </div>
                                )}
                            </div>
                        )}

                        <button
                            type="submit"
                            className="submit-btn"
                            disabled={loading}
                        >
                            {loading ? (
                                <span className="spinner"></span>
                            ) : (
                                isLogin ? 'Sign In' : 'Create Account'
                            )}
                        </button>
                    </form>

                    {/* Toggle Login/Register */}
                    <div className="toggle-auth">
                        <p>
                            {isLogin ? "Don't have an account?" : "Already have an account?"}
                            <button type="button" onClick={toggleMode}>
                                {isLogin ? 'Sign Up' : 'Sign In'}
                            </button>
                        </p>
                    </div>
                </div>

                {/* Decorative Elements */}
                <div className="decorative-circle circle-1"></div>
                <div className="decorative-circle circle-2"></div>
                <div className="decorative-circle circle-3"></div>
            </div>
        </div>
    );
}

export default LoginPage;
