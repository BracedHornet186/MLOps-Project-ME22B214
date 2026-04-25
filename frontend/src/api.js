import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "/api";

// Create an Axios instance
const apiClient = axios.create({
  baseURL: API_URL,
});

// Request interceptor to attach the auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle 401s
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear token and redirect/notify to login
      localStorage.removeItem("access_token");
      // Dispatch a custom event so the App component knows to show the login screen
      window.dispatchEvent(new Event("auth:unauthorized"));
    }
    return Promise.reject(error);
  }
);

export default apiClient;
