import React from 'react';

const ClassificationCard = ({ prediction }) => {
  const { predicted_class: className, probability, top_3_alternatives: alternatives = [] } = prediction;
  const pct = (probability * 100).toFixed(1);

  return (
    <div className="cls-card">
      <div className="cls-lbl">Top prediction</div>
      <div className="cls-name">{className}</div>
      <div className="bar-bg">
        <div className="bar-fill" style={{ width: `${Math.min(pct, 100)}%` }}></div>
      </div>
      <div className="bar-pct">{pct}% confidence</div>
      <hr className="alt-hr" />
      <div className="cls-lbl">Alternatives</div>
      {alternatives.map((alt, idx) => (
        <div key={idx} className="alt-row">
          <span>{alt.class}</span>
          <span className="alt-pct">{(alt.probability * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

export default ClassificationCard;
