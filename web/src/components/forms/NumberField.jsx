import { Input } from "../ui/input";

export function NumberField({ label, value, onChange, step, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field ${hasValue ? "has-value" : ""}`}>
        <Input
          className="md-input"
          type="number"
          step={step}
          value={value ?? ""}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              onChange(null);
              return;
            }
            const n = Number(raw);
            if (Number.isNaN(n)) return;
            onChange(n);
          }}
          disabled={disabled}
          placeholder=" "
        />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}
