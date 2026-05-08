import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './App.css'

const root = document.getElementById('root')
if (!root) {
  throw new Error('missing #root mount node in index.html')
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
