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

import KNNplot from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\KNNplot.png"
import ANNplot from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\ANNplot.png"
import decision_tree2 from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\decision_tree2.png"
import decision_tree3 from "C:\\Users\\Rahul -hp\\Desktop\\seventh sem\\data mining\\2020BTECS00061_LA1\\frontend\\src\\decision_tree3.png"

const Assignment5 = ({file}) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  

  // const [file, setFile] = useState(null);
  const [data, setData] = useState({});
  const [treeImage, setTreeImage] = useState(null);
  const [RegressionData, setRegressionData] = useState(null);
  const [NaiveData, setNaiveData] = useState(null);
  const [KNNData, setKNNData] = useState('');
  const [ANNData, setANNData] = useState('');
  const [resultData, setResultData] = useState([]);

  



  const [showLinear, setshowLinear] = useState(true);
  const [showKNN, setshowKNN] = useState(false);
  const [showANN, setshowANN] = useState(false);
  const [showNaive, setshowNaive] = useState(false);


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
  
  const handleBackend = async (algo) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('algo', algo);

    try {
            const response = await axios.post("http://127.0.0.1:8000/api/regression/", formData, {
                headers: {
                "Content-Type": "multipart/form-data",
                },
            });
            console.log("response = ",response.data) ;
            if(algo === "Linear") {
                setRegressionData(response.data)
            }
            else if(algo === "Naive") {
                console.log("Naive Data ")
                setNaiveData(response.data) ;
            }
            else if(algo === "KNN") {
                console.log("KNN Data ")
                setKNNData(response.data['confusion_matrix']) ;
            }
            else if(algo === "ANN") {
                console.log("ANN Data ")
                setANNData(response.data) ;
            }
          
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

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
        </tbody>
      </table>
    );
  }



  const handleCheckboxClick = (event) => {

    console.log(String(event.target.innerHTML))
  
    if(String(event.target.innerHTML) == "ANN") {
      handleBackend("ANN");
      setshowANN(true); // Toggle the dataVisible state
      setshowKNN(false); // Toggle the dataVisible state
      setshowNaive(false); // Toggle the dataVisible state
      setshowLinear(false); // Toggle the dataVisible state
    }
    else if(String(event.target.innerHTML) == "KNN") {
        handleBackend("KNN");
        setshowANN(false); // Toggle the dataVisible state
        setshowKNN(true); // Toggle the dataVisible state
        setshowNaive(false); // Toggle the dataVisible state
        setshowLinear(false); // Toggle the dataVisible state
        
    }
    else if(String(event.target.innerHTML) == "Linear Regression") {
        handleBackend("Linear");
        setshowANN(false); // Toggle the dataVisible state
        setshowKNN(false); // Toggle the dataVisible state
        setshowNaive(false); // Toggle the dataVisible state
        setshowLinear(true); // Toggle the dataVisible state
        
    }
    else {
        handleBackend("Naive");
        setshowANN(false); // Toggle the dataVisible state
        setshowKNN(false); // Toggle the dataVisible state
        setshowNaive(true); // Toggle the dataVisible state
        setshowLinear(false); // Toggle the dataVisible state
      
    }
  };
  

  return (
    <Box m="20px">
      <Header title="ASSIGNMENT NO 5" subtitle="Regression" />
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
      <b>Linear Regression</b>
      </Button>

      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>Naive Bayes</b>
      </Button>

      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>KNN</b>
      </Button>

      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        onClick={handleCheckboxClick}
      >
      <b>ANN</b>
      </Button>

    

      <br/>
      {/* <input type="checkbox" id="showLinear" name="showLinear" value="Bike" onClick={handleCheckboxClick}/>
      <label for="showLinear"> Show Data: </label><br></br>

      <input type="checkbox" id="showKNN" name="showKNN" value="Bike" onClick={handleCheckboxClickVisual}/>
      <label for="showKNN"> Show Visual Data: </label><br></br> */}

      {showLinear && RegressionData &&
        <>
            <h1>{getTabData(RegressionData)}</h1>
        </>
      }
      
      {showNaive && NaiveData &&
        <>
            <h1>{getTabData(NaiveData)}</h1>
        </>
      }

      {showKNN && KNNData &&
        <>

            <br />
            <img src={KNNplot} alt="" />

            <h1>Iterations: 1</h1>
            <h1>{getTabData(KNNData[0])}</h1>
            <h1>Iterations: 3</h1>
            <h1>{getTabData(KNNData[1])}</h1>
            <h1>Iterations: 5</h1>
            <h1>{getTabData(KNNData[2])}</h1>
            <h1>Iterations: 7</h1>
            <h1>{getTabData(KNNData[3])}</h1>
        </>
      }
      
      {showANN && ANNData &&
        <>
        <br />
        <img src={ANNplot} alt="" />
        <h1>{getTabData(ANNData)}</h1>
        </>
      }
      
    </Box>
  );
};

export default Assignment5;
