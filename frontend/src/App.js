import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import Topbar from "./scenes/global/Topbar";
import Sidebar from "./scenes/global/Sidebar";
import Dashboard from "./scenes/dashboard";
import Assignment1 from "./scenes/assignment1";
import Assignment2 from "./scenes/assignment2";
import Assignment3 from "./scenes/assignment3";
import Assignment5 from "./scenes/assignment5";
import Assignment6 from "./scenes/assignment6";
import Assignment7 from "./scenes/assignment7";
import Assignment72 from "./scenes/assignment72";
import Assignment8 from "./scenes/assignment8";
import Invoices from "./scenes/invoices";
import Bar from "./scenes/bar";
import Form from "./scenes/form";
import Line from "./scenes/line";
import Pie from "./scenes/pie";
import FAQ from "./scenes/faq";
import Geography from "./scenes/geography";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import Calendar from "./scenes/calendar/calendar";

function App() {
  const [theme, colorMode] = useMode();
  const [isSidebar, setIsSidebar] = useState(true);

  const [file, setFile] = useState(null)

  return (
    
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div className="app">
          <Sidebar isSidebar={isSidebar} />
          <main className="content">
            <Topbar setIsSidebar={setIsSidebar} />
            <Routes>
              <Route path="/" element={<Dashboard setfile={setFile} file={file}/>} />
              <Route path="/assignment1" element={<Assignment1 file={file}/>} />
              <Route path="/assignment2" element={<Assignment2 />} />
              <Route path="/assignment3" element={<Assignment3 file={file}/>} />
              <Route path="/assignment5" element={<Assignment5 file={file}/>} />
              <Route path="/assignment6" element={<Assignment6 file={file}/>} />
              <Route path="/assignment7" element={<Assignment7 file={file}/>} />
              <Route path="/assignment72" element={<Assignment72 file={file}/>} />
              <Route path="/assignment8" element={<Assignment8 file={file}/>} />

              <Route path="/invoices" element={<Invoices />} />
              <Route path="/form" element={<Form />} />
              <Route path="/bar" element={<Bar />} />
              <Route path="/pie" element={<Pie />} />
              <Route path="/line" element={<Line />} />
              <Route path="/faq" element={<FAQ />} />
              <Route path="/calendar" element={<Calendar />} />
              <Route path="/geography" element={<Geography />} />
              
            </Routes>
          </main>
        </div>
      </ThemeProvider>
    
  );
}

export default App;
