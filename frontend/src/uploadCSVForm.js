import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import { BarChart} from '@mui/x-charts';
import { Button, Container, Paper } from '@mui/material';
import Plotly from 'react-plotly.js';




const UploadCSVForm = () => {
  const [file, setFile] = useState(null);
  const [data, setData] = useState({});
  const [quantileRow, setquantileRow] = useState(0);
  const [qqRow1, setqqRow1] = useState(0);
  const [qqRow2, setqqRow2] = useState(0);
  const [spRow1, setspRow1] = useState(0);
  const [spRow2, setspRow2] = useState(0);
  const [hgRow, sethgRow] = useState(0);
  const [bpRow, setbpRow] = useState(0);

  const [showData, setshowData] = useState(false);
  const [showVisual, setshowVisual] = useState(false);

  const options = [];
  for (let i = 0; i <= 100; i++) {
    options.push(
      <option key={i} value={i}>
        {i}
      </option>
    );
  }

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

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
            <th>Key</th>
            <th>Mean</th>
            <th>Median</th>
            <th>Mode</th>
            <th>Midrange</th>
            <th>Variance</th>
            <th>Std Deviation</th>
            <th>range</th>
            <th>quartiles</th>
            <th>interquartile_range</th>
            <th>five_number_summary</th>
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
      alert(response.data.statistics)
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

  const handleCheckboxClick = () => {
    console.log(showData)
    setshowData(!showData); // Toggle the dataVisible state
  };
  
  const handleCheckboxClickVisual = () => {
    console.log(showVisual)
    setshowVisual(!showVisual); // Toggle the dataVisible state
  };

  return (
    <div>
      <h2>Upload CSV File</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>

      <br/>
      <input type="checkbox" id="showData" name="showData" value="Bike" onClick={handleCheckboxClick}/>
      <label for="showData"> Show Data: </label><br></br>

      <input type="checkbox" id="showVisual" name="showVisual" value="Bike" onClick={handleCheckboxClickVisual}/>
      <label for="showVisual"> Show Visual Data: </label><br></br>

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
    </div>
  );
};

export default UploadCSVForm;
