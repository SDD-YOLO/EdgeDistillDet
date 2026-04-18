import { M3Select } from "./M3Select";

export function SelectField({ label, value, onChange, options, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field md-field-select ${hasValue ? "has-value" : ""}`}>
        <M3Select value={value} onChange={onChange} options={options} disabled={disabled} ariaLabel={label} />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}
