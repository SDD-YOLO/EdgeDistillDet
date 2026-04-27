export function AdvancedCardsColumn({ cards, renderAdvancedField }) {
  return (
    <div className="config-section">
      {cards.map(({ scope, section }) => (
        <div key={`${scope}-${section.title}`} className="config-card config-card-compact">
          <h3 className="card-header">{section.title}</h3>
          <div className="form-grid training-form-grid">
            {(section.params || []).map((param) => (
              <div key={param.key}>{renderAdvancedField(scope, param)}</div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
