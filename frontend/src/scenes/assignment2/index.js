import React, { useState } from 'react';
import axios from 'axios';
import { Button, Container, TextField, OutlinedInput } from '@mui/material';
import Plot from 'react-plotly.js';
import Header from "../../components/Header";




const CorrelationAnalysis = () => {
  const [file, setFile] = useState(null);
  const [attribute1, setAttribute1] = useState('');
  const [attribute2, setAttribute2] = useState('');
  const [result, setResult] = useState(null);
  const [activeComponent, setActiveComponent] = useState('');
  const [normalizationType, setNormalizationType] = useState('min_max');
  const [decimalScale, setDecimalScale] = useState(2);
  const [scatterData, setScatterData] = useState([]);
  const [columnNames, setColumnNames] = useState([]);



  const handleComponentChange = (componentName) => {
    setActiveComponent(componentName);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    const selectedFile = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      const contents = e.target.result;
      const lines = contents.split('\n');
      if (lines.length > 0) {
        const headerLine = lines[0];
        const columns = headerLine.split(',');
        setColumnNames(columns);
      }
    };
    reader.readAsText(selectedFile);
  };

  const handleAttribute1Change = (event) => {
    setAttribute1(event.target.value);
  };

  const handleAttribute2Change = (event) => {
    setAttribute2(event.target.value);
  };

  const handleNormalizationTypeChange = (event) => {
    setNormalizationType(event.target.value);
    handleComponentChange('normalization'); 
    handleSubmit('http://127.0.0.1:8000/api/perform_normalization/');
  };

  const handleDecimalScaleChange = (event) => {
    setDecimalScale(event.target.value);
    handleSubmit('http://127.0.0.1:8000/api/perform_normalization/');
  };


  const renderTable = (contingencyTable) => {
    const headers = Object.keys(contingencyTable).sort();
    const subHeaders = Object.keys(contingencyTable[headers[0]]).sort();
  
    return (
      <table>
        <thead>
          <tr>
            <th></th>
            {subHeaders.map((subHeader, index) => (
              <th key={index}>{subHeader}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {headers.map((header, index) => (
            <tr key={index}>
              <th>{header}</th>
              {subHeaders.map((subHeader, subIndex) => (
                <td key={subIndex}>{contingencyTable[header][subHeader]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };
  
  
  
  

  const handleSubmit = async (url) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('attribute1', attribute1);
    formData.append('attribute2', attribute2);

    if(normalizationType) formData.append('normalization_type', normalizationType);
    if(decimalScale) formData.append('decimal_scale', decimalScale);

    try {
      const response = await axios.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log(response.data)

      if(response.data.normalized_data) {
        const { sepal_length, sepal_width, petal_length, petal_width } = response.data.normalized_data;
        setScatterData([
            {
              x: sepal_length,
              y: sepal_width,
              mode: 'markers',
              type: 'scatter',
              name: 'Sepal',
              marker: { color: 'blue' }
            },
            {
              x: petal_length,
              y: petal_width,
              mode: 'markers',
              type: 'scatter',
              name: 'Petal',
              marker: { color: 'red' }
            }
          ]) ;
      }

      setResult(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      
        <div style={{marginLeft: "30px"}}>
        <Header title="ASSIGNMENT NO 2" subtitle="Corelation Anaysis Test" />
        </div>

        <div>

        <Button
            variant="contained"
            style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
            component="label">
            <b>Upload File</b>
            <input
            onChange={handleFileChange}
            type="file"
            hidden
            />
        </Button>

        <Button
  variant="contained"
  style={{
    color: "#757ce8",
    backgroundColor: "#e0e0e0",
    marginLeft: "30px",
  }}
  component="label"
>
  <select
    onChange={handleAttribute1Change}
    style={{
      background: "transparent",  // Set transparent background
      outline: "none",            // Remove outline on focus
      border: "none",             // Remove border
    }}
    value={attribute1}
  >
    <option>Select a column</option>
    {columnNames.map((columnName, index) => (
      <option key={columnName} value={columnName}>
        {columnName}
      </option>
    ))}
  </select>

</Button>
       
       <Button
  variant="contained"
  style={{
    color: "#757ce8",
    backgroundColor: "#e0e0e0",
    marginLeft: "30px",
 
  }}
  component="label"
>
  <select
    onChange={handleAttribute2Change}
    style={{
      background: "transparent",  // Set transparent background
      outline: "none",            // Remove outline on focus
      border: "none",             // Remove border
    }}
    value={attribute2}
  >
    <option>Select a column</option>
    {columnNames.map((columnName, index) => (
      <option key={columnName} value={columnName}>
        {columnName}
      </option>
    ))}
  </select>
</Button>

        </div>

        <div style={{ marginLeft: "30px" }}>

        <br />

        <button onClick={() => { handleSubmit('http://127.0.0.1:8000/api/chi2_analyze/'); handleComponentChange('chi2');}} type="submit">Perform Chi-Square Test</button>
        <button onClick={() => {handleSubmit('http://127.0.0.1:8000/api/corelation/'); handleComponentChange('pearson'); }} type="submit">Perform Corelation-Coefficient</button>
        <button onClick={() => {handleComponentChange('normalization'); }} type="submit">
            <select value={normalizationType} onChange={handleNormalizationTypeChange}>
                <option value="min_max">Min-Max Normalization</option>
                <option value="z_score">Z-Score Normalization</option>
                <option value="decimal_scaling">Normalization by Decimal Scaling</option>
            </select>
        </button>

      {result && result.chi2_value && activeComponent === 'chi2' && (
        <div>
          <h3>Results:</h3>
          {/* <p>Contingency Table:</p> */}
          {/* <pre>{JSON.stringify(result.contingency_table, null, 2)}</pre> */}
          <p>Chi-Square Value: {result.chi2_value}</p>
          <p>P-Value: {result.p_value}</p>
          <p>Conclusion: {result.correlation_result}</p>

          <p>Contingency Table: </p>
          {renderTable(result.contingency_table)}

        </div>
      )}

      {result && activeComponent === 'pearson' && (
     
        <div>
          <h3>Results:</h3>
          <p>Correlation Coefficient: {result.correlation_coefficient}</p>
          <p>Covariance: {result.covariance}</p>
          <p>Conclusion: {result.conclusion}</p>
        </div>
      )}
      {activeComponent === 'normalization' && (
        <div>
        {normalizationType === 'decimal_scaling' && (
          <div>

          <br />
            <label>Decimal Scale:</label>
            <input
              type="number"
              value={decimalScale}
              onChange={handleDecimalScaleChange}
              min="1"
              step="1"
            />
          </div>
        )}
        
        <br />
        </div>
      )}


     {result && activeComponent === 'normalization' && <Plot
        data={scatterData}
        layout={{
          title: 'Scatter Plot of Normalized Data',
          xaxis: { title: 'Normalized Length' },
          yaxis: { title: 'Normalized Width' }
        }}
      />
    }

    </div>

    </div>
  );
};

export default CorrelationAnalysis;
