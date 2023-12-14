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

import decision_tree1 from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\decision_tree1.png"
import decision_tree2 from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\decision_tree2.png"
import decision_tree3 from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\decision_tree3.png"

const Assignment3 = ({file}) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  

  // const [file, setFile] = useState(null);
  const [data, setData] = useState({});
  const [treeImage, setTreeImage] = useState(null);
  const [tabData1, settabData1] = useState('');
  const [tabData2, settabData2] = useState('');
  const [tabData3, settabData3] = useState('');
  const [resultData, setResultData] = useState([]);

  



  const [showData, setshowData] = useState(true);
  const [showRules, setshowRules] = useState(false);
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
        <div>
        </div>
    );
  };
  
  const handleUploadTree = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
            const response = await axios.post("http://127.0.0.1:8000/api/info_gain/", formData, {
                headers: {
                "Content-Type": "multipart/form-data",
                },
            });
            console.log("response = ",response.data)

            setResultData(response.data) ;

            settabData1(response.data['entropy'])
            settabData2(response.data['gini'])
            settabData3(response.data['gain'])
            


          
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const [csvData, setCsvData] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  const getTabData = (data) => {

    const confusionMatrix = data['confusion_matrix']

    const TP = confusionMatrix[0][0];
    const FP = confusionMatrix[0][1];
    const FN = confusionMatrix[1][0];
    const TN = confusionMatrix[1][1];
  
    const accuracy = (TP + TN) / (TP + FP + FN + TN);
    const recognitionRate = TP / (TP + FN);
    const misclassificationRate = (FP + FN) / (TP + FP + FN + TN);
    const sensitivity = TP / (TP + FN);
    const specificity = TN / (TN + FP);
    const precision = TP / (TP + FP);
    const recall = sensitivity; // Recall is the same as sensitivity
  
    return MetricsTable({TP,FP,TN,FN,accuracy,recognitionRate,misclassificationRate,sensitivity,specificity,precision,recall, data});
  };
  function MetricsTable({ TP,FP,TN,FN,accuracy,recognitionRate,misclassificationRate,sensitivity,specificity,precision,recall, data }) {
    return (
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>

        <tr>
            <td>TP</td>
            <td>{TP}</td>
          </tr>
          <tr>
            <td>TN</td>
            <td>{TN}</td>
          </tr>
          <tr>
            <td>FP</td>
            <td>{FP}</td>
          </tr>
          <tr>
            <td>FN</td>
            <td>{FN}</td>
          </tr>

          <tr>
            <td>Accuracy</td>
            <td>{accuracy}</td>
          </tr>
          <tr>
            <td>Recognition Rate </td>
            <td>{recognitionRate}</td>
          </tr>
          <tr>
            <td>Misclassification Rate</td>
            <td>{misclassificationRate}</td>
          </tr>
          <tr>
            <td>Sensitivity</td>
            <td>{sensitivity}</td>
          </tr>
          <tr>
            <td>Specificity</td>
            <td>{specificity}</td>
          </tr>
          <tr>
            <td>Precision</td>
            <td>{precision}</td>
          </tr>
          <tr>
            <td>Recall</td>
            <td>{recall}</td>
          </tr>
          <tr>
            <td>Coverage</td>
            <td>{data['coverage']}</td>
          </tr>
          <tr>
            <td>Toughness</td>
            <td>{data['toughness']}</td>
          </tr>
        </tbody>
      </table>
    );
  }
  function getRuleData( data ) {

    console.log(data) ;

    return (
      <table>
        <thead>
          <tr>
            <th>Rules</th>
          </tr>
        </thead>
        <tbody>
        
          {data['rules'].map((val, index) => (
            <tr key={index}>
              <td>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

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
      handleUploadTree();

      new Promise(resolve => setTimeout(resolve, 2000));
    
  
    if(String(event.target.innerHTML) == "Decision Tree") {
      console.log(treeImage)
      setshowData(true); // Toggle the dataVisible state
      setshowVisual(false); // Toggle the dataVisible state
    }
    else if(String(event.target.innerHTML) == "Rules") {
      console.log(treeImage)
      setshowData(false); // Toggle the dataVisible state
      setshowVisual(false); // Toggle the dataVisible state
      setshowRules(true); // Toggle the dataVisible state
    }
    else {
      console.log(treeImage)
      setshowData(false); // Toggle the dataVisible state
      setshowVisual(true); // Toggle the dataVisible state
      setshowRules(false);
    }
  };
  

  return (
    <Box m="20px">
      <Header title="ASSIGNMENT NO 3" subtitle="Decision Tree" />
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
      <b>Decision Tree</b>
      </Button>


      <Button
        variant="contained"
        id="resultdata"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>Rules</b>
      </Button>

      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>Tabulate</b>
      </Button>

    

      <br/>
      {/* <input type="checkbox" id="showData" name="showData" value="Bike" onClick={handleCheckboxClick}/>
      <label for="showData"> Show Data: </label><br></br>

      <input type="checkbox" id="showVisual" name="showVisual" value="Bike" onClick={handleCheckboxClickVisual}/>
      <label for="showVisual"> Show Visual Data: </label><br></br> */}

      {showData && 
    <div>
      {/* <h2>Accuracy: {resultData.accuracy}</h2>
      <h2>Confusion Matrix:</h2> */}
      {/* <table> */}
        {/* <tbody> */}
          {/* {resultData.confusion_matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Measure: {resultData.measure}</h2>
      <h2>Precision: {resultData.precision}</h2>
      <h2>Recall: {resultData.recall}</h2> */}

      {/* Display the image */}
      <h1>Entropy</h1>
      {<img src={decision_tree1} alt="Decision Tree" />}
      <h1>Gain Ratio</h1>
      {<img src={decision_tree2} alt="Decision Tree" />}
      <h1>Gini Index</h1>
      {<img src={decision_tree3} alt="Decision Tree" />}
    </div>
      }
      {showRules && <div id='visuals'>
      <h2>Data Rules</h2>
      
        <h1>Entropy</h1>
        {getRuleData(tabData1)}
        <h1>Gain Ratio</h1>
        {getRuleData(tabData2)}
        <h1>Gini Index</h1>
        {getRuleData(tabData3)}
      
    </div>}

      {showVisual && <div id='visuals'>
      <h2>Data Visualization</h2>
      
        <h1>Entropy</h1>
        {getTabData(tabData1)}
        <h1>Gain Ratio</h1>
        {getTabData(tabData2)}
        <h1>Gini Index</h1>
        {getTabData(tabData3)}
      
    </div>}
    </Box>
  );
};

export default Assignment3;
