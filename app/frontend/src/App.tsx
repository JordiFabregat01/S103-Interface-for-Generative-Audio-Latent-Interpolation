import { useState } from "react";
import ExplorerView from "./components/ExplorerView";
import WorkspaceView from "./components/WorkspaceView";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState<"explorer" | "workspace">(
    "explorer"
  );

  return (
    <div>
      <div className="topbar">
        <button
          className={activeTab === "explorer" ? "tab active" : "tab"}
          onClick={() => setActiveTab("explorer")}
        >
          Explorer
        </button>

        <button
          className={activeTab === "workspace" ? "tab active" : "tab"}
          onClick={() => setActiveTab("workspace")}
        >
          Workspace
        </button>
      </div>

      {activeTab === "explorer" && <ExplorerView />}
      {activeTab === "workspace" && <WorkspaceView />}
    </div>
  );
}

export default App;