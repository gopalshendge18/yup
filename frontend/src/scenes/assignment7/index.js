import React, { useState } from 'react';

const Assignment7 = () => {
  const [confidence, setConfidence] = useState(0.7);
  const [support, setSupport] = useState(0.5);
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/apriori', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ confidence, support }),
      });

      if (!response.ok) {
        throw new Error('Error fetching data');
      }

      const data = await response.json();
      const parsedRules = JSON.parse(data.rules || '[]'); // Parse the rules string into an array
      setRules(parsedRules);
    } catch (error) {
      console.error('Error:', error.message);
      setError('Error fetching data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFetchData = () => {
    fetchData();
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <div style={{ textAlign: 'center', marginTop: '50px' }}>
        <label >Confidence:</label>
        <input
          type="number"
          value={confidence}  // Keep it dynamic
          onChange={(e) => setConfidence(e.target.value)}
        />
      </div>
      <div style={{  marginTop: '30px' , marginLeft:'8px'}}>
        <label>Support:</label>
        <input
          type="number"
          value={support}  // Keep it dynamic
          onChange={(e) => setSupport(e.target.value)}
        />
      </div>
      <button style={{  marginTop: '10px' , marginLeft:'8px'}} onClick={handleFetchData} disabled={loading}>
        Fetch Association Rules
      </button>
      <div>
        {loading && <p>Loading...</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {rules.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Antecedents</th>
                <th>Consequents</th>
                <th>Antecedent Support</th>
                <th>Consequent Support</th>
                <th>Support</th>
                <th>Confidence</th>
                <th>Lift</th>
                {/* Add more columns based on your data */}
              </tr>
            </thead>
            <tbody>
              {rules.map((rule, index) => (
                <tr key={index}>
                  <td>{rule.antecedents.join(',')}</td>
                  <td>{rule.consequents.join(',')}</td>
                  <td>{rule['antecedent support']}</td>
                  <td>{rule['consequent support']}</td>
                  <td>{rule.support}</td>
                  <td>{rule.confidence}</td>
                  <td>{rule.lift}</td>
                  {/* Add more cells based on your data */}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Assignment7;
