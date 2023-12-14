import React, { useState, useEffect } from 'react';

const ClusterButtons = ({ clusterType, onButtonClick, selectedCluster }) => (
    <button
        onClick={() => onButtonClick(clusterType)}
        style={{
            backgroundColor: selectedCluster === clusterType ? 'darkslategray' : 'gray',
            color: 'white',
            padding: '10px',
            borderRadius: '5px',
            cursor: 'pointer',
            margin: '5px',
        }}
    >
        {clusterType}
    </button>
);

const Assignment6 = () => {
    const [imagePaths, setImagePaths] = useState({
        agnes: null,
        DIANA: null,
        kmeans: null,
        pam: null,
        birch: null,
        dbscan: null,
    });
    const [selectedClusters, setSelectedClusters] = useState([]);
    const [tabulatedResults, setTabulatedResults] = useState([]);

    useEffect(() => {
        // Fetch the image paths from Django API when clusters are selected
        const fetchImagePaths = async () => {
            const paths = await Promise.all(
                selectedClusters.map(async clusterType => {
                    const response = await fetch(`http://127.0.0.1:8000/${clusterType.toLowerCase()}`);
                    const data = await response.json();
                    return { clusterType, imagePath: data.image_path };
                })
            );

            const updatedImagePaths = paths.reduce((acc, { clusterType, imagePath }) => {
                acc[clusterType] = imagePath;
                return acc;
            }, { ...imagePaths });

            setImagePaths(updatedImagePaths);
        };

        if (selectedClusters.length > 0) {
            fetchImagePaths();
        }
    }, [selectedClusters]);

    const handleButtonClick = clusterType => {
        setSelectedClusters(prevSelectedClusters => {
            if (prevSelectedClusters.includes(clusterType)) {
                return prevSelectedClusters.filter(type => type !== clusterType);
            } else {
                return [...prevSelectedClusters, clusterType];
            }
        });
    };

    const handleTabulateResults = async () => {
        const response = await fetch('http://127.0.0.1:8000/acc');
        const data = await response.json();
        
        // Parse the "results" string into a JavaScript array
        const parsedResults = JSON.parse(data.results);
    
        console.log('Tabulated Results:', parsedResults);
        setTabulatedResults(parsedResults);
    };
    

    return (
        <div style={{ textAlign: 'center', marginTop: '50px' }}>
            {['agnes', 'kmeans', 'pam', 'birch', 'dbscan'].map(clusterType => (
                <ClusterButtons
                    key={clusterType}
                    clusterType={clusterType}
                    onButtonClick={handleButtonClick}
                    selectedCluster={selectedClusters.includes(clusterType) ? clusterType : ''}
                />
            ))}
            <button onClick={handleTabulateResults} style={{ margin: '10px' }}>
                Tabulate Results
            </button>

            {/* Display all the images */}
            <div style={{ marginTop: '20px', color: 'darkslategray', display: 'flex', justifyContent: 'center', flexWrap: 'wrap' }}>
                {selectedClusters.map(clusterType => (
                    <div key={clusterType} style={{ margin: '10px' }}>
                        <p style={{ color: 'darkslategray' }}>
                            Image loaded after {clusterType} clustering:
                        </p>
                        <img
                            src={`http://127.0.0.1:8000${imagePaths[clusterType]}`}
                            alt={`${clusterType} Clustering`}
                            style={{ maxWidth: '100%', height: 'auto' }}
                        />
                    </div>
                ))}
            </div>

            {/* Display tabulated results */}
            <div style={{ marginTop: '20px', color: 'darkslategray' }}>
    <p>Tabulated Results:</p>
    <table style={{ width: '70%', margin: 'auto', borderCollapse: 'collapse', fontSize: '18px', border: '1px solid darkslategray' }}>
        <thead>
            <tr style={{ background: 'darkslategray', color: 'white' }}>
                <th style={{ padding: '10px', border: '1px solid darkslategray' }}>Algorithm</th>
                <th style={{ padding: '10px', border: '1px solid darkslategray' }}>ARI Score</th>
            </tr>
        </thead>
        <tbody>
            {tabulatedResults.map(result => (
                <tr key={result.Algorithm} style={{ textAlign: 'center', border: '1px solid darkslategray' }}>
                    <td style={{ padding: '10px', border: '1px solid darkslategray' }}>{result.Algorithm}</td>
                    <td style={{ padding: '10px', border: '1px solid darkslategray' }}>{result['ARI Score']}</td>
                </tr>
            ))}
        </tbody>
    </table>
</div>
</div> 
);
};

export default Assignment6;
