import React, { useState, useRef, useEffect, useCallback } from "react";
import { OperatorExecutionButton } from "@fiftyone/operators";
import { useTheme } from "@mui/material";

interface OperatorButtonWithWarningProps {
  operatorUri: string;
  executionParams: Record<string, any>;
  onOptionSelected?: (option: any) => void;
  onSuccess?: (response: any) => void;
  onError?: (error: any) => void;
  disabled?: boolean;
  variant?: string;
  color?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

/**
 * Wrapper around OperatorExecutionButton that detects empty dropdowns
 * and shows a helpful warning message.
 *
 * This monitors for the Menu component to appear after clicking, and if
 * the menu appears but has no items (empty), it shows a warning.
 */
export function OperatorButtonWithWarning({
  operatorUri,
  executionParams,
  onOptionSelected,
  onSuccess,
  onError,
  disabled,
  variant,
  color,
  style,
  children,
}: OperatorButtonWithWarningProps) {
  const theme = useTheme();
  const [showNoTargetsWarning, setShowNoTargetsWarning] = useState(false);
  const [isCheckingMenu, setIsCheckingMenu] = useState(false);
  const checkTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Monitor the DOM for menu appearance and check if it's empty
  useEffect(() => {
    if (!isCheckingMenu) return;

    const checkForEmptyMenu = () => {
      // Look for the MUI Menu/Popover component
      const menus = document.querySelectorAll('.MuiPopover-root, [role="presentation"]');

      for (const menu of Array.from(menus)) {
        // Check if menu is visible
        const rect = (menu as HTMLElement).getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          // Menu is visible - check if it has any menu items
          const menuItems = menu.querySelectorAll('[role="menuitem"], .MuiMenuItem-root');

          if (menuItems.length === 0) {
            // Menu is open but has no items!
            setShowNoTargetsWarning(true);
            setIsCheckingMenu(false);

            // Close the empty menu by simulating an outside click
            const backdrop = menu.querySelector('.MuiBackdrop-root');
            if (backdrop) {
              (backdrop as HTMLElement).click();
            }
            return;
          } else {
            // Menu has items, all good
            setIsCheckingMenu(false);
            return;
          }
        }
      }

      // No menu found yet, keep checking for a bit longer
      if (checkTimeoutRef.current) {
        checkTimeoutRef.current = setTimeout(checkForEmptyMenu, 50);
      }
    };

    // Start checking after a short delay
    checkTimeoutRef.current = setTimeout(checkForEmptyMenu, 100);

    // Stop checking after 500ms total
    const stopTimeout = setTimeout(() => {
      setIsCheckingMenu(false);
      if (checkTimeoutRef.current) {
        clearTimeout(checkTimeoutRef.current);
        checkTimeoutRef.current = null;
      }
    }, 500);

    return () => {
      if (checkTimeoutRef.current) {
        clearTimeout(checkTimeoutRef.current);
      }
      clearTimeout(stopTimeout);
    };
  }, [isCheckingMenu]);

  // Wrap the onOptionSelected to detect successful option selection
  const handleOptionSelected = useCallback((option: any) => {
    // If an option was selected, hide the warning
    setShowNoTargetsWarning(false);
    if (onOptionSelected) {
      onOptionSelected(option);
    }
  }, [onOptionSelected]);

  // Intercept clicks on the button wrapper
  const handleWrapperClick = useCallback(() => {
    if (!disabled) {
      setIsCheckingMenu(true);
    }
  }, [disabled]);

  const isDark = theme.palette.mode === "dark";

  return (
    <div>
      {showNoTargetsWarning && (
        <div
          style={{
            padding: "12px 16px",
            backgroundColor: isDark ? "rgba(255, 152, 0, 0.1)" : "#fff3e0",
            borderLeft: "4px solid #ff9800",
            borderRadius: "4px",
            marginBottom: "16px",
            color: theme.palette.text.primary,
          }}
        >
          <strong>⚠️ Delegated Execution Required</strong>
          <p
            style={{
              margin: "8px 0 0 0",
              fontSize: "13px",
              color: theme.palette.text.secondary,
              lineHeight: "1.5",
            }}
          >
            You need delegated operators to run this operation. Please configure an orchestrator to enable this feature.
          </p>
        </div>
      )}
      <div onClick={handleWrapperClick}>
        <OperatorExecutionButton
          operatorUri={operatorUri}
          executionParams={executionParams}
          onOptionSelected={handleOptionSelected}
          onSuccess={onSuccess}
          onError={onError}
          disabled={disabled}
          variant={variant}
          color={color}
          style={style}
        >
          {children}
        </OperatorExecutionButton>
      </div>
    </div>
  );
}
