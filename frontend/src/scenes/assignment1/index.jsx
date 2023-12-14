import { Box, Typography, useTheme } from "@mui/material";
import { FormThemeProvider } from 'react-form-component'
import { tokens } from "../../theme";
import AdminPanelSettingsOutlinedIcon from "@mui/icons-material/AdminPanelSettingsOutlined";
import LockOpenOutlinedIcon from "@mui/icons-material/LockOpenOutlined";
import SecurityOutlinedIcon from "@mui/icons-material/SecurityOutlined";
import Header from "../../components/Header";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MuiFileInput } from 'mui-file-input'
import { BarChart} from '@mui/x-charts';
import { Button, Container, TextField, OutlinedInput } from '@mui/material';
import Plotly from 'react-plotly.js';

const Assignment1 = ({file}) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const columns = [
    { field: "id", headerName: "ID" },
    {
      field: "name",
      headerName: "Name",
      flex: 1,
      cellClassName: "name-column--cell",
    },
    {
      field: "age",
      headerName: "Age",
      type: "number",
      headerAlign: "left",
      align: "left",
    },
    {
      field: "phone",
      headerName: "Phone Number",
      flex: 1,
    },
    {
      field: "email",
      headerName: "Email",
      flex: 1,
    },
    {
      field: "accessLevel",
      headerName: "Access Level",
      flex: 1,
      renderCell: ({ row: { access } }) => {
        return (
          <Box
            width="60%"
            m="0 auto"
            p="5px"
            display="flex"
            justifyContent="center"
            backgroundColor={
              access === "admin"
                ? colors.greenAccent[600]
                : access === "manager"
                ? colors.greenAccent[700]
                : colors.greenAccent[700]
            }
            borderRadius="4px"
          >
            {access === "admin" && <AdminPanelSettingsOutlinedIcon />}
            {access === "manager" && <SecurityOutlinedIcon />}
            {access === "user" && <LockOpenOutlinedIcon />}
            <Typography color={colors.grey[100]} sx={{ ml: "5px" }}>
              {access}
            </Typography>
          </Box>
        );
      },
    },
  ];

  // const [file, setFile] = useState(null);
  const [data, setData] = useState({});
  const [quantileRow, setquantileRow] = useState(0);
  const [qqRow1, setqqRow1] = useState(0);
  const [qqRow2, setqqRow2] = useState(0);
  const [spRow1, setspRow1] = useState(0);
  const [spRow2, setspRow2] = useState(0);
  const [hgRow, sethgRow] = useState(0);
  const [bpRow, setbpRow] = useState(0);

  const [showData, setshowData] = useState(true);
  const [showVisual, setshowVisual] = useState(false);

  const options = [];
  for (let i = 0; i <= 100; i++) {
    options.push(
      <option key={i} value={i}>
        {i}
      </option>
    );
  }

  // const handleFileChange = (e) => {
  //   setFile(e.target.files[0]);
  // };

  const renderValue = (value) => {
    if (Array.isArray(value)) {
      return value.join(', ');
    } else {
      return value;
    }
  };


  const renderTable = () => {
    return (
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Mean</th>
            <th>Median</th>
            <th>Mode</th>
            <th>Midrange</th>
            <th>Variance</th>
            <th>SD</th>
            <th>range</th>
            <th>quartiles</th>
            <th>IQR</th>
            <th>FNS</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([key, values]) => (
              <tr>
              <td>{key}</td>
              {Object.entries(values).map(([key1, values1]) => (
                  <td className="fixed-width-cell">{renderValue(values1)}</td>
                ))}
              </tr>
          ))}
        </tbody>
      </table>
    );
  };
  
  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    if (file) {
      const parsedData = await parseCSV(file);
      setCsvData(parsedData.data);
      setIsDataLoaded(true);
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/upload-csv/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Content-Disposition': `attachment; filename="${file.name}"`,
        },
      });
      // alert(response.data.statistics)
      console.log('File uploaded:', response.data);
      setData(response.data.statistics)
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const [csvData, setCsvData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const parsedData = await parseCSV(file);
      setCsvData(parsedData.data);
      setIsDataLoaded(true);
    }
  };

  const parseCSV = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target.result;
        const parsedData = result.split('\n').map((row) => row.split(','));
        resolve({ data: parsedData });
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsText(file);
    });
  };

  const handleCheckboxClick = (event) => {
    handleUpload();
  
    if(String(event.target.innerHTML) == "Get Result") {
      setshowData(true); // Toggle the dataVisible state
      setshowVisual(false); // Toggle the dataVisible state
    }
    else {
      setshowData(false); // Toggle the dataVisible state
      setshowVisual(true); // Toggle the dataVisible state
    }
  };
  

  return (
    <Box m="20px">
      <Header title="ASSIGNMENT NO 1" subtitle="Data Visualization and Statistics" />
      <h2>Upload CSV File</h2>
      {/* <Button
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
      </Button> */}

      {/* <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px" }}
        onClick={handleUpload}
      >
      <b>Upload</b>
      </Button> */}


      <Button
        variant="contained"
        id="resultdata"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>Get Result</b>
      </Button>

      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>Visualize</b>
      </Button>

    

      <br/>
      {/* <input type="checkbox" id="showData" name="showData" value="Bike" onClick={handleCheckboxClick}/>
      <label for="showData"> Show Data: </label><br></br>

      <input type="checkbox" id="showVisual" name="showVisual" value="Bike" onClick={handleCheckboxClickVisual}/>
      <label for="showVisual"> Show Visual Data: </label><br></br> */}

      {showData && <div id='data_table'>
        <h1>Data Table</h1>
          {Object.keys(data).length > 0 ? renderTable() : <p>Loading...</p>}
        </div>
      }
      {showVisual && <div id='visuals'>
      <h2>Data Visualization</h2>
      {isDataLoaded && (
        <div>
          {/* Quantile Plot */}
          <div className='center'>
          <select value={quantileRow} onChange={(event) => {setquantileRow(event.target.value)}}>
            <option value="">Select...</option>
            {options}
          </select>
          </div>
          
          <Plotly
            data={[
              {
                x: csvData.map((row) => row[quantileRow]),
                type: 'scatter',
                mode: 'markers',
                name: 'Quantile Plot',
              },
            ]}
            layout={{ title: 'Quantile Plot' }}
          />
          <br></br>
          {/* Quantile-Quantile (Q-Q) Plot */}
          <div className='center'>
            <select value={qqRow1} onChange={(event) => {setqqRow1(event.target.value)}}>
              <option value="">Select...</option>
              {options}
            </select>
            <select style={{marginLeft: "50px"}} value={qqRow2} onChange={(event) => {setqqRow2(event.target.value)}}>
              <option value="">Select...</option>
              {options}
            </select>
            </div>
          <Plotly
            data={[
              {
                x: csvData.map((row) => row[qqRow1]),
                y: csvData.map((row) => row[qqRow2]),
                type: 'scatter',
                mode: 'markers',
                name: 'Q-Q Plot',
              },
            ]}
            layout={{ title: 'Q-Q Plot' }}
          />

          <br></br>
          {/* Histogram */}
          <div className='center'>
          <select className='center' value={hgRow} onChange={(event) => {sethgRow(event.target.value)}}>
            <option value="">Select...</option>
            {options}
          </select>
          </div>
          <Plotly
            data={[
              {
                x: csvData.map((row) => row[hgRow]),
                type: 'histogram',
                name: 'Histogram',
              },
            ]}
            layout={{ title: 'Histogram' }}
          />

          <br></br>
          {/* Scatter Plot */}
          <div className='center'>
          <select value={spRow1} onChange={(event) => {setspRow1(event.target.value)}}>
            <option value="">Select...</option>
            {options}
          </select>
          <select  value={spRow2} onChange={(event) => {setspRow2(event.target.value)}}>
            <option value="">Select...</option>
            {options}
          </select>
          </div>
          <Plotly
            data={[
              {
                x: csvData.map((row) => row[spRow1]),
                y: csvData.map((row) => row[spRow2]),
                type: 'scatter',
                mode: 'markers',
                name: 'Scatter Plot',
              },
            ]}
            layout={{ title: 'Scatter Plot' }}
          />

          <br></br>
          {/* Box Plot */}
          <div className='center'>
          <select className='center' value={bpRow} onChange={(event) => {setbpRow(event.target.value)}}>
            <option value="">Select...</option>
            {options}
          </select>
          </div>
          <Plotly
            data={[
              {
                y: csvData.map((row) => row[bpRow]),
                type: 'box',
                name: 'Box Plot',
              },
            ]}
            layout={{ title: 'Box Plot' }}
          />
        </div>
      )}
    </div>}
    </Box>
  );
};

export default Assignment1;
