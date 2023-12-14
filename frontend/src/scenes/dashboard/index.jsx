import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import Header from "../../components/Header";
import LineChart from "../../components/LineChart";
import GeographyChart from "../../components/GeographyChart";
import BarChart from "../../components/BarChart";
import StatBox from "../../components/StatBox";
import ProgressCircle from "../../components/ProgressCircle";
import Papa from "papaparse";
import { useState, useEffect } from "react";

const Dashboard = ({setfile, file}) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [data, setData] = useState([])

  useEffect(()=>{
    if(file) {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function (results) {
          setData(results.data); // Store the parsed data in state
        },
      });
    }
  },[])

  const handleFileChange = (e) => {

    const file = e.target.files[0] ;

    setfile(e.target.files[0]);

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: function (results) {
        setData(results.data); // Store the parsed data in state
      },
    });

  };

  return (
    <>
    <Box m="20px">
      <Header title="Dashboard" subtitle="Data Mining" />
      <h2>Upload CSV File</h2>
      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        component="label"
      >
        <b>Upload File</b>
        <input
          onChange={handleFileChange}
          type="file"
          hidden
        />
      </Button>
      
    </Box>

    <div>
        {data.length > 0 && (
          <table>
            <thead>
              <tr>
                {Object.keys(data[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((value, index) => (
                    <td key={index}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    
    </>
  );
};

export default Dashboard;
