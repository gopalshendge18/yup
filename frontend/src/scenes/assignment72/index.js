import React, { useState, useEffect } from 'react';

const RuleList = ({ rules, selectedMethod }) => {
  const rawRules = rules[selectedMethod] || '[]'; // Use '[]' as default in case the rules are not available
  const parsedRules = JSON.parse(rawRules);

  console.log('Filtered Rules:', rawRules); // Log filtered rules

  
  return (
    <div>
      <h2>Filtered Rules</h2>
      <table>
        <thead>
          <tr>
            <th>Antecedents</th>
            <th>Consequents</th>
            <th>Confidence</th>
            <th>Support</th>
            <th>Lift</th>
            {/* Add other table headers as needed */}
          </tr>
        </thead>
        <tbody>
          {parsedRules.map((rule, index) => (
            <tr key={index}>
              <td>{rule.antecedents.join(', ')}</td>
              <td>{rule.consequents.join(', ')}</td>
              <td>{rule.confidence}</td>
              <td>{rule.support}</td>
              <td>{rule.lift}</td>
              {/* Add other table cells as needed */}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};


const Assignment72 = () => {
  const [confidence, setConfidence] = useState('');
  const [support, setSupport] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('chi');
  const [rules, setRules] = useState({});

  const fetchData = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/rules`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          confidence: parseFloat(confidence),
          support: parseFloat(support),
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setRules(data);

      console.log('Fetched Rules:', data); // Log fetched rules
    } catch (error) {
      console.error('Error fetching rules:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, [confidence, support]);

  const handleSubmit = event => {
    event.preventDefault();
    fetchData();
  };

  return (
    <div>
      <h1>Rule Page</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Confidence:
          <input
            type="text"
            value={confidence}
            onChange={e => setConfidence(e.target.value)}
          />
        </label>
        <br />
        <label>
          Support:
          <input
            type="text"
            value={support}
            onChange={e => setSupport(e.target.value)}
          />
        </label>
        <br />
        <label>
          Method:
          <select value={selectedMethod} onChange={e => setSelectedMethod(e.target.value)}>
            <option value="rules_cosine">Cosine</option>
            <option value="rules_kulczynski">Kulczynski</option>
            <option value="rules_all_confidence">All Confidence</option>
            <option value="rules_max_confidence">Max Confidence</option>
            <option value="rules_lift">Lift</option>
          </select>
        </label>
        <br />
        <button type="submit">Submit</button>
      </form>
      <RuleList rules={rules} selectedMethod={selectedMethod} />
    </div>
  );
};

export default Assignment72;
