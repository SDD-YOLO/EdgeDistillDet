import { AdvancedCardsColumn } from "./AdvancedCardsColumn";

export function AdvancedViewContainer({ advancedCardsLeft, advancedCardsRight, renderAdvancedField }) {
  return (
    <div className="panel-grid">
      <AdvancedCardsColumn cards={advancedCardsLeft} renderAdvancedField={renderAdvancedField} />
      <AdvancedCardsColumn cards={advancedCardsRight} renderAdvancedField={renderAdvancedField} />
    </div>
  );
}