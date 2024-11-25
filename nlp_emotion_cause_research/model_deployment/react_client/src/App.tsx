import { Home } from './views/Home'
import { Routes, Route } from 'react-router-dom'

import Join from './components/Join/Join'
import Chat from './components/Chat/Chat'

export function App() {
  return (
    <Routes>
      <Route path="/" Component={Join} />
      <Route path="/chat" Component={Chat} />
    </Routes>
  )
}

export default App
